# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT pretraining runner."""

import argparse
import json
# import loggerplus as logger
import logging
import math
import multiprocessing
import numpy as np
import os
import random
import signal
import time
import warnings
import torch

# try:
#     import kfac
# except:
#     pass

from optimizers.lamb import LAMBOptimizer as FusedLAMB
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import src.modeling as modeling

from src.dataset import ShardedPretrainingDataset, DistributedSampler
from src.schedulers import PolyWarmUpScheduler, LinearWarmUpScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.tokenization import get_wordpiece_tokenizer, get_bpe_tokenizer
from src.utils import is_main_process, get_world_size, get_rank

import optimizers.mkor as mkor
# import optimizers.kaisa as kaisa
import optimizers.eva as eva
import optimizers.backend as eva_backend
import utils.comm as comm
from torch.optim import SGD, Adam

try:
    from torch.cuda.amp import autocast, GradScaler

    TORCH_FP16 = True
except:
    TORCH_FP16 = False

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

logger = logging.getLogger(__name__
                           )
# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True


signal.signal(signal.SIGTERM, signal_handler)


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, masked_lm_labels,
                seq_relationship_score=None, next_sentence_labels=None):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size),
                                      masked_lm_labels.view(-1))
        if seq_relationship_score is not None and next_sentence_labels is not None:
            next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2),
                                              next_sentence_labels.view(-1))
            return masked_lm_loss + next_sentence_loss
        return masked_lm_loss


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Optional json config to override defaults below
    parser.add_argument("--config_file", default=None, type=str,
                        help="JSON config for overriding defaults")

    ## Required parameters. Note they can be provided in the json
    parser.add_argument("--input_dir", default=None, type=str,
                        help="The input data dir containing .hdf5 files for the task.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output dir for checkpoints and logging.")
    parser.add_argument("--model_ckpt_dir", default=None, type=str,
                        help="The input dir for checkpoints.")
    parser.add_argument("--model_config_file", default=None, type=str,
                        help="The BERT model config")

    ## Dynamic Masking Parameters
    parser.add_argument("--masked_token_fraction", type=float, default=0.2,
                        help='Fraction of tokens to mask per sequence')
    parser.add_argument("--max_predictions_per_seq", type=int, default=80,
                        help='Maximum masked tokens per sequence')

    ## Training Configuration
    parser.add_argument('--disable_progress_bar', default=False, action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--num_steps_per_checkpoint', type=int, default=100,
                        help="Number of update steps between writing checkpoints.")
    parser.add_argument('--skip_checkpoint', default=False, action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--checkpoint_activations', default=False, action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument('--log_prefix', type=str, default='logfile',
                        help='Prefix for log files. This is just the prefix of '
                             'name and should not contain directories')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Use PyTorch AMP training")
    parser.add_argument('--prune_inputs', default=False, type=bool,
                        help="Prune inputs to linear layers")

    ## Hyperparameters
    parser.add_argument("--optimizer", default="lamb", type=str,
                        help="The name of the optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--lr_decay", default='poly', type=str,
                        choices=['poly', 'linear', 'cosine'],
                        help="Learning rate decay type.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Proportion of training to perform linear learning rate "
                             "warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--global_batch_size", default=2 ** 16, type=int,
                        help="Global batch size for training.")
    parser.add_argument("--local_batch_size", default=8, type=int,
                        help="Per-GPU batch size for training.")
    parser.add_argument("--max_steps", default=1000, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--previous_phase_end_step", default=0, type=int,
                        help="Final step of previous phase")
    parser.add_argument("--additional", default='', type=str,
                        help="additional text")

    ## KFAC Hyperparameters
    parser.add_argument('--kfac', default=False, action='store_true',
                        help='Use KFAC')
    parser.add_argument('--kfac_inv_interval', type=int, default=10,
                        help='iters between kfac inv ops')
    parser.add_argument('--kfac_factor_interval', type=int, default=1,
                        help='iters between kfac cov ops')
    parser.add_argument('--kfac_stat_decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation')
    parser.add_argument('--kfac_damping', type=float, default=0.001,
                        help='KFAC damping factor')
    parser.add_argument('--kfac_kl_clip', type=float, default=0.001,
                        help='KFAC KL gradient clip')
    parser.add_argument('--kfac_skip_layers', nargs='+', type=str,
                        default=['BertLMPredictionHead', 'embedding'],
                        help='Modules to ignore registering with KFAC '
                             '(default: [BertLMPredictionHead, embedding])')

    # Set by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()

    # Hacky way to figure out to distinguish arguments that were found
    # in sys.argv[1:]
    aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for arg in vars(args):
        aux_parser.add_argument('--' + arg)
    cli_args, _ = aux_parser.parse_known_args()

    # Argument precedent: cli_args > config_file > argparse defaults
    if args.config_file is not None:
        with open(args.config_file) as jf:
            configs = json.load(jf)
        for key in configs:
            if key in vars(args) and key not in vars(cli_args):
                setattr(args, key, configs[key])

    return args


def setup_training(args):
    assert (torch.cuda.is_available())

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args.model_output_dir = os.path.join(args.output_dir, 'pretrain_ckpts')
    if is_main_process():
        os.makedirs(args.model_output_dir, exist_ok=True)

    # logger.init(
    #     handlers=[
    #         logger.StreamHandler(verbose=is_main_process()),
    #         logger.FileHandler(
    #             os.path.join(args.output_dir, args.log_prefix + '.txt'),
    #             overwrite=False, verbose=is_main_process()),
    #         logger.TorchTensorboardHandler(
    #             os.path.join(args.output_dir, 'tensorboard'),
    #             verbose=is_main_process()),
    #         logger.CSVHandler(
    #             os.path.join(args.output_dir, args.log_prefix + '_metrics.csv'),
    #             overwrite=False, verbose=is_main_process()),
    #     ]
    # )

    logger.info('Torch distributed initialized (world_size={}, backend={})'.format(
        get_world_size(), torch.distributed.get_backend()))

    if not TORCH_FP16 and args.fp16:
        raise ValueError('FP16 training enabled but unable to import torch.cuda.amp.'
                         'Is the torch version >= 1.6?')

    if args.global_batch_size % get_world_size() != 0 and is_main_process():
        warnings.warn('global_batch_size={} is not divisible by world_size={}.'
                      ' The last batch will be padded with additional '
                      'samples.'.format(
            args.global_batch_size, get_world_size()))
    args.local_accumulated_batch_size = math.ceil(
        args.global_batch_size / get_world_size())

    if args.local_accumulated_batch_size % get_world_size() != 0 and is_main_process():
        warnings.warn('local_accumulated_batch_size={} is not divisible '
                      'by local_batch_size={}. local_accumulated_batch_size '
                      'is global_batch_size // world_size. The last '
                      'batch will be padded with additional samples'.format(
            args.local_accumulated_batch_size, get_world_size()))
    args.accumulation_steps = math.ceil(
        args.local_accumulated_batch_size / args.local_batch_size)

    return args

def remap_parameters(model_dict):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'intermediate.dense' in k or 'pooler.dense' in k:
            new_k = k.replace('dense', 'dense_act')
        elif 'transform.dense' in k:
            new_k = k.replace('dense', 'dense_act')
        else:
            new_k = k
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict

def prepare_model(args):
    config = modeling.BertConfig.from_json_file(args.model_config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)

    if args.prune_inputs:
        pass

    checkpoint = None
    global_steps = 0
    args.resume_step = 0
    checkpoint_names = [f for f in os.listdir(args.model_ckpt_dir)
                        if f.endswith(".pt")]
    if len(checkpoint_names) > 0:
        args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip())
                                for x in checkpoint_names])

        checkpoint = torch.load(
            os.path.join(args.model_ckpt_dir, "ckpt_{}.pt".format(args.resume_step)),
            map_location="cpu"
        )
        new_checkpoint = remap_parameters(checkpoint['model'])
        model.load_state_dict(new_checkpoint, strict=True)
        # print("+++++++++++++++++++++++++")
        print("load!")

        if args.previous_phase_end_step > args.resume_step:
            raise ValueError('previous_phase_end_step={} cannot be larger '
                             'than resume_step={}'.format(
                args.previous_phase_end_step, args.resume_step))
        global_steps = args.resume_step - args.previous_phase_end_step

        logger.info('Resume from step {} checkpoint'.format(args.resume_step))

    model.to(args.device)
    model.checkpoint_activations(args.checkpoint_activations)

    model = DDP(model, device_ids=[args.local_rank])

    criterion = BertPretrainingCriterion(config.vocab_size)

    return model, checkpoint, global_steps, criterion, args


# def set_no_decay(model, optimizer):
#     no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
#     no_decay_param_group = {"params": [], "weight_decay": 0.0}
#     for key, value in optimizer.param_groups[0].items():
#         if key not in ["params", "weight_decay"]:
#             no_decay_param_group[key] = value
#     params = optimizer.param_groups[0]["params"]
#     for name, param in model.named_parameters():
#         if any(nd in name for nd in no_decay):
#             no_decay_param_group["params"].append(param)
#             for i in range(len(params)):
#                 if param.shape == params[i].shape and torch.all(param == params[i]):
#                     del params[i]
#                     break
#     optimizer.param_groups.append(no_decay_param_group)
#     if hasattr(optimizer, 'optimizer'):
#         optimizer.optimizer.param_groups.append(no_decay_param_group)
#     return optimizer


def get_optimizer(grouped_parameters, lr, model, optimizer_name, weight_decay=0.01):
    if optimizer_name == 'eva':
        eva_backend.init("Torch")
        lamb_parameters = []
        eva_parameters = []
        lamb_names = []
        eva_names = []
        supported_modules = ['Linear','Conv2d','LinearActivation']
        # sgd_layers = [module for module in model.modules() if
        #               isinstance(module, torch.nn.Linear) and module.out_features == 30528]
        # print([[module.__class__.__name__,len([mc for mc in module.named_children()])] for module in model.modules()])
        # print([p for p in model.parameters()])
        for module in model.modules():
            # print(module)
            # print(len([mc for mc in module.named_children()]),module.__class__.__name__)
            if module.__class__.__name__ in supported_modules:
                e_p = {"params":module.parameters()}
                eva_parameters.append(e_p)
                # print("eva!\n")
            elif len([mc for mc in module.named_children()]) == 0 and module.__class__.__name__ != "Identity":
                l_p1 = {"params":module.parameters()}
                lamb_parameters.append(l_p1)
                # print(module)
                # print("lamb!\n")
            else:
                # print("skip!\n")
                continue
        # backend = comm.get_comm_backend()
        # optimizer = eva.Eva(model, lr=lr, sgd_layers=sgd_layers,
        #                         optimizer=FusedLAMB(grouped_parameters, lr=lr),
        #                     )
        # the repository use lamb here
        optimizer = eva.Eva(model, lr=lr, eva_parameters=eva_parameters,lamb_parameters=lamb_parameters,
                                optimizer=SGD(eva_parameters,lr=lr),optimizer1=FusedLAMB(lamb_parameters, lr=0.0005, betas=(0.86,0.975))
                            )
        return optimizer
    elif optimizer_name == 'lamb':
        return FusedLAMB(grouped_parameters, lr=lr, betas=(0.86,0.975))
    elif optimizer_name == 'mkor':
        sgd_layers = [module for module in model.modules() if
                      isinstance(module, torch.nn.Linear) and module.out_features == 30528]
        # backend = comm.get_comm_backend()
        optimizer = mkor.MKOROptimizer(model, lr=lr, weight_decay=weight_decay, inv_freq=10, measure_time=False,
                                       svd=False,
                                    #    backend=backend, 
                                       sgd_layers=sgd_layers, grad_accum_steps=args.accumulation_steps,
                                       half_precision=True,
                                       optimizer=FusedLAMB(grouped_parameters, lr=lr),
                                       )
        return optimizer
    # elif optimizer_name == 'kaisa':
    #     optimizer = kaisa.KAISAOptimizer(model)
    #     return optimizer
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(grouped_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(grouped_parameters, lr=lr, weight_decay=weight_decay)


def prepare_optimizers(args, model, checkpoint, global_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if args.lr_decay == 'poly':
        Scheduler = PolyWarmUpScheduler
    elif args.lr_decay == 'linear':
        Scheduler = LinearWarmUpScheduler
    elif args.lr_decay == 'cosine':
        Scheduler = CosineAnnealingLR
    else:
        raise ValueError('Unknown lr decay "{}"'.format(args.lr_decay))

    optimizer = get_optimizer(optimizer_grouped_parameters, args.learning_rate, model, args.optimizer,
                              args.weight_decay)

    if checkpoint is not None:
        try:
            if args.resume_step >= args.previous_phase_end_step:
                if args.optimizer != "mkor":
                    keys = list(checkpoint['optimizer']['state'].keys())
                    # Override hyperparameters from previous checkpoint
                    for key in keys:
                        checkpoint['optimizer']['state'][key]['step'] = global_steps
                    for i, item in enumerate(checkpoint['optimizer']['param_groups']):
                        checkpoint['optimizer']['param_groups'][i]['step'] = global_steps
                        checkpoint['optimizer']['param_groups'][i]['t_total'] = args.max_steps
                        checkpoint['optimizer']['param_groups'][i]['warmup'] = args.warmup_proportion
                        checkpoint['optimizer']['param_groups'][i]['lr'] = args.learning_rate
                else:
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['step'] = global_steps
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['t_total'] = args.max_steps
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['warmup'] = args.warmup_proportion
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['lr'] = args.learning_rate
        except:
            pass
        try:
            if args.optimizer == "lamb" and "optimizer" in checkpoint["optimizer"]:
                print("Loaded LAMB Optimizer from MKOR Checkpoint")
                optimizer.load_state_dict(checkpoint['optimizer']['optimizer'])
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.grad_accum_steps = args.accumulation_steps
            if hasattr(optimizer, 'accumulated_steps'):
                for i in range(len(optimizer.accumulated_steps)):
                    optimizer.accumulated_steps[i] = 0
        except:
            optimizer = get_optimizer(optimizer_grouped_parameters, args.learning_rate, model, args.optimizer,
                              args.weight_decay)
            for param_group in optimizer.param_groups:
                param_group['step'] = global_steps

    
    if args.optimizer == 'eva':
        try:
            lr_schedulers = [Scheduler(optimizer.optimizer, warmup=args.warmup_proportion,
                                       total_steps=32000)]
        except:
            lr_schedulers = [Scheduler(optimizer.optimizer, T_max=32000, eta_min=args.learning_rate * 1e-4)]
        
        lr_scheduler_1 = CosineAnnealingLR(optimizer.optimizer1, T_max=32000, eta_min=0.0005 * 1e-4) 
        lr_schedulers.append(lr_scheduler_1)
    else:
        try:
            lr_schedulers = [Scheduler(optimizer, warmup=args.warmup_proportion,
                                    total_steps=32000)]
        except:
            lr_schedulers = [Scheduler(optimizer, T_max=32000, eta_min=args.learning_rate * 1e-4)]

    scaler = None
    if args.fp16:
        scaler = GradScaler()
        if checkpoint is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        if hasattr(optimizer, 'update_grad_scale'):
            optimizer.update_grad_scale(scaler.get_scale())

    preconditioner = None
    if args.kfac:
        # preconditioner = kfac.KFAC(
        #     model,
        #     lr=args.learning_rate,
        #     factor_decay=args.kfac_stat_decay,
        #     damping=args.kfac_damping,
        #     kl_clip=args.kfac_kl_clip,
        #     factor_update_freq=args.kfac_factor_interval,
        #     inv_update_freq=args.kfac_inv_interval,
        #     # Skip TrainingHeads which contains the decoder, a Linear module
        #     # with shape (seq_len, vocab_size), such that it is too large to invert
        #     skip_layers=args.kfac_skip_layers,
        #     # BERT calls KFAC very infrequently so no need to optimize for
        #     # communication. Optimize for memory instead.
        #     comm_method=kfac.CommMethod.MEM_OPT,
        #     # Compute the factors and update the running averages during the
        #     # forward backward pass b/c we are using grad accumulation but
        #     # not accumulating the input/output data
        #     accumulate_data=False,
        #     compute_factor_in_hook=True,
        #     distribute_layer_factors=False,
        #     # grad_scaler=scaler,
        # )

        lrs = Scheduler(preconditioner, warmup=args.warmup_proportion,
                        total_steps=32000)
        lr_schedulers.append(lrs)

        if checkpoint is not None and 'preconditioner' in checkpoint:
            preconditioner.load_state_dict(checkpoint['preconditioner'])

        if is_main_process():
            logger.info(preconditioner)

    return optimizer, preconditioner, lr_schedulers, scaler


def prepare_dataset(args, checkpoint):
    input_files = []
    if os.path.isfile(args.input_dir):
        input_files.append(args.input_dir)
    elif os.path.isdir(args.input_dir):
        for path in Path(args.input_dir).rglob('*.hdf5'):
            if path.is_file():
                input_files.append(str(path))

    with open(args.model_config_file) as f:
        configs = json.load(f)
        vocab_size = configs['vocab_size']
        vocab_file = configs['vocab_file']
        lowercase = configs['lowercase']
        tokenizer = configs['tokenizer']
    mask_token_id = None
    if tokenizer == 'wordpiece':
        tokenizer = get_wordpiece_tokenizer(vocab_file, uppercase=not lowercase)
    elif tokenizer == 'bpe':
        tokenizer = get_bpe_tokenizer(vocab_file, uppercase=not lowercase)
    else:
        raise ValueError('Unknown tokenizer \'{}\'. Options are '
                         '\'wordpiece\' and \'bpe\''.format(tokenizer))
    mask_token_id = tokenizer.token_to_id('[MASK]')

    dataset = ShardedPretrainingDataset(input_files, mask_token_id,
                                        args.max_predictions_per_seq, args.masked_token_fraction,
                                        vocab_size=vocab_size)
    sampler = DistributedSampler(dataset, get_world_size(), rank=get_rank())

    if checkpoint is not None and 'sampler' in checkpoint:
        sampler.load_state_dict(checkpoint['sampler'])

    loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                         batch_size=args.local_batch_size, num_workers=4, pin_memory=True)

    if is_main_process():
        logger.info('Samples in dataset: {}'.format(len(dataset)))
        logger.info('Samples per device: {}'.format(len(sampler)))
        logger.info('Sampler starting index: {}'.format(sampler.index))
        logger.info('Batches in dataloader: {}'.format(len(loader)))
    return loader, sampler


def take_optimizer_step(optimizer, preconditioner, model, scaler):
    # torch.nn.utils.clip_grad_norm_(model.parameters(),2.0,2)
    if preconditioner is not None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        preconditioner.step()
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
        if hasattr(optimizer, "update_grad_scale"):
            optimizer.update_grad_scale(scaler.get_scale())
    else:
        optimizer.step()
    # print("Memory Usage: ", torch.cuda.max_memory_allocated(device=model.device) / (2.0 ** 30))

    try:
        optimizer.zero_grad()
        if hasattr(optimizer, "optimizer"):
            optimizer.optimizer.zero_grad()
        if hasattr(optimizer, "optimizer1"):
            optimizer.optimizer1.zero_grad()
    except:
        for param in model.parameters():
            param.grad = None


def forward_backward_pass(model, criterion, scaler, batch, divisor,
                          sync_grads=True):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    if scaler is not None:
        with autocast():
            # if is_main_process():
            #     if torch.any(torch.isnan(input_ids)):
            #         print('input_ids is nan')
            #     if torch.any(torch.isnan(segment_ids)):
            #         print('segment_ids is nan')
            #     if torch.any(torch.isnan(input_mask)):
            #         print('input_mask is nan')
            prediction_scores, seq_relationship_score = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask)
            # if is_main_process():
            #     if torch.isnan(prediction_scores).any():
            #         print('prediction_scores is nan')
            #     if torch.any(torch.isnan(seq_relationship_score)):
            #         print('seq_relationship_score is nan')
            #     for module in model.modules():
            #         if isinstance(module, torch.nn.Linear):
            #             if torch.any(torch.isnan(module.weight)):
            #                 print('weight is nan')
            #             if module.bias is not None:
            #                 if torch.any(torch.isnan(module.bias)):
            #                     print('bias is nan')
            #                     break
            loss = criterion(
                prediction_scores,
                masked_lm_labels,
                seq_relationship_score,
                next_sentence_labels)
    else:
        prediction_scores, seq_relationship_score = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask)
        loss = criterion(
            prediction_scores,
            masked_lm_labels,
            seq_relationship_score,
            next_sentence_labels)

    loss = loss / divisor

    if not sync_grads:
        with model.no_sync():
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
    else:
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    return loss


def main(args):
    global timeout_sent

    model, checkpoint, global_steps, criterion, args = prepare_model(args)
    optimizer, preconditioner, lr_schedulers, scaler = prepare_optimizers(
        args, model, checkpoint, global_steps)
    dataloader, datasampler = prepare_dataset(args, checkpoint)

    model.train()
    most_recent_ckpts_paths = []
    average_loss = None
    current_loss = 0.
    epoch = 0
    training_steps = 0
    train_time_start = time.time()
    ignore_time = True

    if checkpoint is not None and 'epoch' in checkpoint:
        epoch = checkpoint['epoch']

    # Note: We loop infinitely over epochs, termination is handled
    #       via iteration count
    while True:
        datasampler.set_epoch(epoch)
        if not args.disable_progress_bar:
            train_iter = tqdm(dataloader, disable=not is_main_process())
        else:
            train_iter = dataloader

        for batch in train_iter:
            sync_grads = False
            if args.accumulation_steps == 1 or (training_steps > 0
                                                and training_steps % args.accumulation_steps == args.accumulation_steps - 1):
                sync_grads = True

            batch = [t.to(args.device) for t in batch]
            loss = forward_backward_pass(model, criterion, scaler, batch,
                                         args.accumulation_steps, sync_grads=sync_grads)
            if ignore_time:
                modeling.timer.zero_time()
                ignore_time = False
            else:
                if is_main_process() and modeling.timer.measure:
                    modeling.timer.save("results", "BERT Forward Pass Time")

            current_loss += loss.item()


            if sync_grads:
                take_optimizer_step(optimizer, preconditioner,
                                    model, scaler)
                for lrs in lr_schedulers:
                    lrs.step()
                global_steps += 1
                if average_loss is None:
                    average_loss = current_loss
                else:
                    average_loss = 0.95 * average_loss + 0.05 * current_loss
                # logger.log(tag='train',
                #            step=global_steps + args.previous_phase_end_step,
                #            epoch=epoch,
                #            average_loss=average_loss,
                #            step_loss=current_loss,
                #            learning_rate=optimizer.param_groups[0]['lr'])
                if is_main_process():
                    wandb.log(
                           {"step":global_steps + args.previous_phase_end_step,
                           "epoch":epoch,
                           "average_loss":average_loss,
                           "step_loss":current_loss,
                           "learning_rate":optimizer.param_groups[0]['lr']})
                    if args.optimizer == 'eva':
                        wandb.log(
                            {'vg_sum':optimizer.vg_sum,
                            "damping":optimizer.damping,
                            "kl_clip":optimizer.kl_clip})
                current_loss = 0.

            if (global_steps >= args.max_steps or timeout_sent or
                    (training_steps > 0 and training_steps % (
                            args.num_steps_per_checkpoint * args.accumulation_steps
                    ) == 0)):
                if global_steps >= args.max_steps or timeout_sent:
                    final_time = time.time() - train_time_start
                if is_main_process() and not args.skip_checkpoint:
                    # Save a trained model
                    logger.info('Saving checkpoint: global_steps={}'.format(
                        global_steps + args.previous_phase_end_step))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_save_file = os.path.join(
                        args.model_output_dir,
                        "ckpt_{}.pt".format(global_steps + args.previous_phase_end_step))
                    if args.optimizer != 'eva':
                        model_dict = {
                            'model': model_to_save.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'sampler': datasampler.state_dict(),
                            'epoch': epoch,
                        }
                    else:
                        model_dict = {
                            'model': model_to_save.state_dict(),
                            'sampler': datasampler.state_dict(),
                            'epoch': epoch,
                        }
                    if preconditioner is not None:
                        model_dict['preconditioner'] = preconditioner.state_dict()
                    if scaler is not None:
                        model_dict['scaler'] = scaler.state_dict()
                    torch.save(model_dict, output_save_file)

                    most_recent_ckpts_paths.append(output_save_file)
                    # if len(most_recent_ckpts_paths) > 3:
                    #     ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                    #     os.remove(ckpt_to_be_removed)

            # Exiting the training due to hitting max steps, or being sent a
            # timeout from the cluster scheduler
            if global_steps >= args.max_steps or timeout_sent:
                return global_steps, final_time

            training_steps += 1

        epoch += 1


if __name__ == "__main__":
    args = parse_arguments()

    if args.input_dir is None:
        raise ValueError('--input_dir must be provided via arguments or the '
                         'config file')
    if args.output_dir is None:
        raise ValueError('--output_dir must be provided via arguments or the '
                         'config file')
    if args.model_config_file is None:
        raise ValueError('--model_config_file must be provided via arguments '
                         'or the config file')

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    args = setup_training(args)
    if is_main_process():
        import wandb
        wandb.init(project="bert_pytorch",name=args.optimizer + str(args.learning_rate),notes=args.additional,mode='offline')
    logger.info('TRAINING CONFIG: {}'.format(args))
    with open(args.model_config_file) as f:
        logger.info('MODEL CONFIG: {}'.format(json.load(f)))

    start_time = time.time()
    global_steps, train_time = main(args)
    runtime = time.time() - start_time

    logger.info("runtime: {}  train_time: {}  training_seq_per_sec: {}".format(
        runtime, train_time,
        args.global_batch_size * global_steps / train_time))

