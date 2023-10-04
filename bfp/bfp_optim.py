# Copyright (c) 2021, Parallel Systems Architecture Laboratory (PARSA), EPFL & 
# Machine Learning and Optimization Laboratory (MLO), EPFL. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the PARSA, EPFL & MLO, EPFL
#    nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import math
from .bfp_ops import float_to_bfp_blocked, unpack_bfp_args
from torch.optim.optimizer import Optimizer

class BFPAdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.bfp_args = {'num_format': 'bfp',
                'sparsity_num_format': 'bfp',
                'rounding_mode': 'stoc',
                'epsilon': 1e-8,
                'mant_bits': 3,
                'bfp_block_size': 64,
                'weight_mant_bits': 15,
                'in_sparsity': False,
                'w_sparsity': False,
                'grad_sparsity': False,
                'N': [2],
                'M': [4],
                'rearrange': False,
                'sparsity_frac': 0,
                'device': 'cuda'}

        super(BFPAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BFPAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if self.bfp_args['num_format'] == 'fp32':
                    #print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;optimizer fp32')
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    updated_value = p
                elif self.bfp_args['num_format'] == 'bfp':
                    #print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;optimizer bfp')
                    updated_value = float_to_bfp_blocked(p.addcdiv_(exp_avg, denom, value=-step_size), sgd_update=True, **self.bfp_args)

                p.copy_(updated_value)
                #p.addcdiv_(exp_avg, denom, value=-step_size)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



_bfp_optims = {}
def _gen_bfp_optim(optim, name):
    class BFPOptim(optim):
        """
        Wrap the model's original optimizer in BFP

        Perform the original optimizer's  update function in fp32
        Convert the weights to two BFP formats: One with wide and another with narrow mantissas.
            Wide weights are used in future weight updates
            Narrow weights are used in forward and backward passes.
        """
        def __init__(self, *args, **kwargs):
            self.bfp_args = unpack_bfp_args(kwargs)
            super().__init__(*args, **kwargs)

        def step(self, *args, **kwargs):
            if self.bfp_args['num_format'] == 'fp32':
                return super().step(*args, **kwargs)

            # Move wide weights to p.data
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    # Init step, just constraint pdata
                    if 'shadow_p' not in state:
                        p.data.copy_(float_to_bfp_blocked(p.data, sgd_update=True, **self.bfp_args))
                    else:
                        shadow_p = state['shadow_p']
                        p.data.copy_(shadow_p)

            # Apply step
            loss = super().step(*args, **kwargs)

            # Move wide weights to shadow_p and move extracted weights to p.data
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'shadow_p' not in state:
                        state['shadow_p'] = torch.zeros_like(p.data)

                    shadow_p = state['shadow_p']
                    shadow_p.copy_(float_to_bfp_blocked(p.data, sgd_update=True, **self.bfp_args))
                    p.data.copy_(float_to_bfp_blocked(p.data, **self.bfp_args))

            return loss

    BFPOptim.__name__ = "BFP" + name
    return BFPOptim


def get_bfp_optim(optim, name):
    if name not in _bfp_optims:
        _bfp_optims[name] = _gen_bfp_optim(optim, name)

    return _bfp_optims[name]
