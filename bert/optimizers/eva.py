import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import optimizers.backend as backend

from optimizers.eva_utils import get_vector_a, get_vector_g
import logging
logger = logging.getLogger()


def clip_norm_(mat, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(mat) == 0:
        return torch.tensor(0.)
    device = mat.device
    total_norm = torch.norm(mat, norm_type).to(device)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    mat.mul_(clip_coef_clamped.to(device))
    return total_norm

def get_damping(step):
    # max_damping = 0.03
    # min_damping = 0.005
    # damping = min_damping + (max_damping - min_damping) * step/32000
    # return damping
    return 0.03

# class Eva(optim.Optimizer):
class Eva():
    """Accelerate Distributed K-FAC with Sublinear Memory Cost
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.03)
      kl_clip (float): clipping parameter for gradient scaling (kl_clip > 0: kl-clip, kl_clip = 0: re-scale, kl-clip < 0: None)
      factor_decay (float): running average coefficient for KVs
      exclude_vocabulary_size: exclude the pre-softmax linear layer in the Transformer
      hook_enabled (bool): enable the hook events to save the immediate states (a and g)
      exclude_parts='': exclude CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor for time breakdowns
    """
    def __init__(self,
                 model,
                 optimizer,
                 optimizer1,
                 eva_parameters,
                 lamb_parameters,
                 lr=0.1,
                 damping=0.3,
                 fac_update_freq=1,
                 kfac_update_freq=1,
                 kfac_batch_size=16,
                 kl_clip=0.001,
                 factor_decay=0.9,
                 exclude_vocabulary_size=30528,
                 hook_enabled=True,
                 exclude_parts='',
                 grad_scale=1.0
                 ):

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq) 

        # super(Eva, self).__init__([], defaults)

        self.lr = lr
        self.damping = damping
        self.fac_update_freq = fac_update_freq
        self.kfac_update_freq = kfac_update_freq

        # self.sgd_layers = sgd_layers
        self.lamb_parameters = lamb_parameters
        self.eva_parameters = eva_parameters
        self.grad_scale = grad_scale

        self.fac_update_freq = fac_update_freq
        self.kfac_batch_size = kfac_batch_size
        self.kl_clip = kl_clip if (kl_clip is not None and kl_clip >= 0) else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_a, self.m_g = {}, {}
        self.handles = []
        
        # scheduling results
        self.module_ranks = None

        self.steps = 0
        self.optimizer = optimizer
        self.optimizer1 = optimizer1
        self.param_groups = optimizer.param_groups
        self.param_groups1 = optimizer1.param_groups
        ''' TODO
            设定好optimizer的param_groups类，与run_pretraining.py中进行对接，还要保证lr_scheduler的正常使用
        '''
        self.param_groups[0]['step'] = 0
        self.param_groups1[0]['step'] = 0
        self.vg_sum = 0

        self.optimizer_base = optimizer1

        self.accu_a = {}
        self.a_cnt = {}
        self.accu_g = {}
        self.g_cnt = {}

    def update_grad_scale(self, scaler):
        self.grad_scale = scaler
        # print("Getting Scaler: ", scaler)

    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_a(input[0].data[0:self.kfac_batch_size], module).to(dtype=torch.float32)
                if module not in self.m_a:
                    self.m_a[module] = torch.zeros(new.size()).to('cuda')
                    self.a_cnt[module] = 1
                    self.accu_a[module] = new
                else:
                    self.accu_a[module] = self.accu_a[module] + new
                    if self.a_cnt[module] == 7:
                        self.m_a[module].mul_(1-self.factor_decay).add_(self.accu_a[module], alpha=self.factor_decay)
                        self.accu_a[module] = new
                        self.a_cnt[module] = 1
                    else:
                        self.a_cnt[module] += 1
                        self.accu_a[module] = self.accu_a[module] + new
            if backend.comm.size() > 1:
                self.handles.append(backend.comm.allreduce_async_(self.m_a[module], op=backend.comm.Average))

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_g(grad_output[0].data[0:self.kfac_batch_size], module).to(dtype=torch.float32) / self.grad_scale
                if module not in self.m_g:
                    self.m_g[module] = torch.zeros(new.size()).to('cuda')
                    self.g_cnt[module] = 1
                    self.accu_g[module] = new
                else:
                    self.accu_g[module] = self.accu_g[module] + new
                    if self.g_cnt[module] == 7:
                        self.m_g[module].mul_(1-self.factor_decay).add_(self.accu_g[module], alpha=self.factor_decay)
                        self.accu_g[module] = new
                        self.g_cnt[module] = 1
                    else:
                        self.g_cnt[module] += 1
                        self.accu_g[module] = self.accu_g[module] + new
            if backend.comm.size() > 1:
                self.handles.append(backend.comm.allreduce_async_(self.m_g[module], op=backend.comm.Average))

    def _register_module_hooks(self, model):
        """Register forard/backward hooks to supported modules"""
        supported_modules = {'Linear', 'Conv2d', 'LinearActivation'}
        name_idx = 0
        for module in model.modules():
            # if module in self.sgd_layers:
            #     continue
            classname = module.__class__.__name__
            if classname in supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module.register_forward_pre_hook(self._forward_hook_event)
                module.register_backward_hook(self._backward_hook_event)  # used in pytorch1.4, and pytorch1.8 (full_backward_hook is not fired when its grad_input is None)
                #module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.10
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1
        if backend.comm.rank() == 0:
            logger.info("#register modules: %s", len(self.modules))

	### Precondition gradients
    def _precondition_grads(self):
        """Compute preconditioned gradients via Eva"""
        g_sum = 0
        v_sum = 0
        vg_sum = 0

        for module in self.modules:
            # get ma, mg, grad
            ma = self.m_a[module].view(-1, 1).to(torch.float)
            mg = self.m_g[module].view(-1, 1).to(torch.float)
            grad = self._get_grad(module)
            
            #if backend.comm.rank() == 0:
            #    logger.info("mg: %s" % (mg))
            
            # compute intermediate states
            a = (ma.T @ ma).item()
            g = (mg.T @ mg).item()
            ag = (mg.T @ grad @ ma).item()
            
            #if backend.comm.rank() == 0 and self.steps % 60 == 0:
            #    logger.info("a: %f, g: %f, ag: %f" % (a, g, ag))
            #    logger.info("beta: %f", ag/(a * g + self.damping))

            # compute preconditioned grads
            v = (mg @ ma.T).mul_(-ag/(a * g + self.damping))
            # clip_norm_(v,2.0)
            v.add_(grad)
            v.div_(self.damping)

            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                # transform preconditioned gradient into gradient scale
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                        vg_sum += (bias * module.bias.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        v_sum += (bias * bias).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                        g_sum += (module.bias.grad.data * module.bias.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
            del v
        self.vg_sum = vg_sum
        # scale preconditioned gradient
        if self.kl_clip is not None:
            if self.kl_clip > 0: # kl-clip
                nu = min(1.0, math.sqrt(self.kl_clip / vg_sum)) if vg_sum > 0 else 1.0
            else: # re-scale
                nu = math.sqrt(g_sum / v_sum)

            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)

    def _get_grad(self, module):
        """Get gradient with shape [output_dim, input_dim] for module"""
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        grad = grad.to(dtype=torch.float32)
        return grad    

    def zero_grad(self):
        pass
        
    ### Perform one K-FAC step
    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = get_damping(self.steps)
        self.fac_update_freq = self.fac_update_freq
        self.kfac_update_freq = self.kfac_update_freq

        if self.steps % self.fac_update_freq == 0 and backend.comm.size() > 1:
            for handle in self.handles:
                backend.comm.synchronize(handle)
            self.handles = []
        
        self._precondition_grads()

        self.optimizer.step()
        self.optimizer1.step()
        self.steps += 1
        self.param_groups[0]['step'] += 1
        self.param_groups1[0]['step'] += 1
