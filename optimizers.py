import math
import torch
from torch.optim.optimizer import Optimizer
from compressors import Compressor

class CSGD(Optimizer):
    """
    Compressed SGD optimizer: applies compression to gradients using a Compressor
    before performing SGD updates (including momentum, weight decay, nesterov).
    """
    def __init__(self,
                 compressor: Compressor,
                 lr: float = 0.01,
                 momentum: float = 0.9,
                 dampening: float = 0,
                 weight_decay: float = 0.01,
                 nesterov: bool = False):
        if not isinstance(compressor, Compressor):
            raise TypeError("compressor must be an instance of Compressor")
        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        super().__init__(compressor.model.parameters(), defaults)
        self.compressor = compressor
        # map parameter tensor to its named key for compression
        self._param_to_name = {p: name for name, p in compressor.model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with compressed gradients.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                # compress gradient via compressor
                name = self._param_to_name.get(p)
                if name is None:
                    raise KeyError(f"Parameter not found in compressor mapping: {p}")
                d_p = self.compressor.compress(name, p)
                # apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                # apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    buf = param_state.get('momentum_buffer')
                    if buf is None:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                # update parameter
                p.add_(d_p, alpha=-lr)

        return loss

class CAdamW(Optimizer):
    """
    Compressed AdamW optimizer: compresses gradients via Compressor before AdamW update
    (including bias correction and decoupled weight decay, supports amsgrad).
    """
    def __init__(self,
                 compressor: Compressor,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 amsgrad: bool = False):
        if not isinstance(compressor, Compressor):
            raise TypeError("compressor must be an instance of Compressor")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(compressor.model.parameters(), defaults)
        self.compressor = compressor
        self._param_to_name = {p: name for name, p in compressor.model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                name = self._param_to_name.get(p)
                if name is None:
                    raise KeyError(f"Parameter not found in compressor mapping: {p}")
                grad = self.compressor.compress(name, p)

                state = self.state[p]
                # increment step
                step = state.get('step', 0) + 1
                state['step'] = step

                # initialize state
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # use max for amsgrad
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(eps)

                bias_correction1 = 1 - beta1 ** step
                step_size = lr / bias_correction1

                # decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss