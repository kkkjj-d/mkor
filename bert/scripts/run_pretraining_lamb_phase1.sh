#!/bin/bash

# Change to 2 for Phase 2 training
PHASE=1
PROC_PER_NODE=8
NODES=1
MASTER_RANK=0

if [[ "$PHASE" -eq 1 ]]; then
        CONFIG=config/bert_pretraining_phase2_config.json
        DATA=/home/yanxindong/mkor/bert/data/encoded/sequences_lowercase_max_seq_len_512_next_seq_task_true
else
        CONFIG=config/bert_kfac_pretraining_phase2_config.json
        DATA=/home/yanxindong/mkor/bert/data/encoded/sequences_lowercase_max_seq_len_512_next_seq_task_true
fi

mkdir -p logs


# PHASE 1
# mpirun -hostfile $HOSTFILE -np $NODES -ppn 1  bash scripts/launch_pretraining.sh  \
#     --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
#     --kwargs \
#     --input_dir $DATA \
#     --output_dir results/bert_pretraining_6e-3-wd0.01-wu2843-inv50-4kbatchsize \
#     --config_file $CONFIG \
#     --weight_decay 0.01 \
#     --num_steps_per_checkpoint 200

#    --lr_decay cosine \
# PHASE 2
bash scripts/launch_pretraining.sh  \
   --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
    --kwargs \
   --config_file $CONFIG \
   --input_dir $DATA \
   --output_dir results/eva \
   --model_ckpt_dir "/home/yanxindong/mkor/bert/data/ckpt_dir_mlperf"  \
   --weight_decay 0.01 \
   --num_steps_per_checkpoint 800 \
   --additional $1
