from math import ceil
from matplotlib import pyplot as plt
import torch
from descent import gradient_descent, mirror_descent, gradient_descent_full, mirror_descent_full

# Generic configurable compressor to select strategy, error correction, and update task
class Compressor:
    def __init__(self, model, k, strategy='TopK', error_correction='none', update_task=None, update_kwargs=None):
        self.model = model
        self.k = k
        self.strategy = strategy
        self.error_correction = error_correction
        self.update_task = update_task
        self.update_kwargs = update_kwargs or {}
        self.w = {}
        self.e = {}
        self.g = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.w[name] = torch.ones_like(param)
            if self.error_correction == 'EF':
                self.e[name] = torch.zeros_like(param)
            elif self.error_correction == 'EF21':
                self.g[name] = torch.zeros_like(param)

    def skip(self, name):
        return 'ln' in name

    def update(self, batch, lr, eta, num_steps):
        if not self.update_task:
            return
        # full update across all parameters
        if self.update_task in ('mirror_descent_full', 'gradient_descent_full'):
            update_fn = {
                'mirror_descent_full': mirror_descent_full,
                'gradient_descent_full': gradient_descent_full
            }[self.update_task]
            # call full-impact update
            # include error buffer if EF
            full_kwargs = dict(self.update_kwargs)
            if self.error_correction == 'EF':
                full_kwargs['errors'] = self.e
            self.w = update_fn(
                self.model,
                batch,
                self.w,
                lr,
                eta,
                **full_kwargs,
                num_steps=num_steps
            )
            return
        # per-parameter update
        update_fn = {
            'mirror_descent': mirror_descent,
            'gradient_descent': gradient_descent
        }[self.update_task]
        for name, param in self.model.named_parameters():
            if self.skip(name):
                continue
            impact = self.w[name]
            # call per-parameter update with possible EF buffer
            per_kwargs = dict(self.update_kwargs)
            if self.error_correction == 'EF':
                per_kwargs['error'] = self.e[name]
            self.w[name] = update_fn(
                self.model,
                batch,
                name,
                impact,
                lr,
                eta,
                **per_kwargs,
                num_steps=num_steps
            )

    def compress(self, name, param):
        if self.skip(name):
            return torch.zeros_like(param.grad)
        k = ceil(self.k * param.numel())
        grad = param.grad
        if self.error_correction == 'EF':
            grad = grad + self.e[name]
        elif self.error_correction == 'EF21':
            grad = grad - self.g[name]

        # apply strategy
        if self.strategy == 'TopK':
            flat = grad.view(-1)
            topk_vals, topk_idx = flat.abs().topk(k)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(0, topk_idx, True)
            comp = mask.view(param.grad.size()) * grad
        elif self.strategy == 'ImpK':
            weighted_grad = grad * self.w[name]
            flat = weighted_grad.view(-1)
            topk_vals, topk_idx = flat.abs().topk(k)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(0, topk_idx, True)
            comp_flat = flat * mask
            comp_flat = comp_flat * (k / self.w[name].view(-1)[topk_idx].sum())
            comp = comp_flat.view(param.grad.size())
        elif self.strategy == 'SCAM':
            weighted_grad = grad * self.w[name]
            flat = weighted_grad.view(-1)
            topk_vals, topk_idx = flat.abs().topk(k)
            mask = torch.zeros_like(flat)
            mask[topk_idx] = self.w[name].view(-1)[topk_idx]
            mask = mask * (k / mask.sum())
            comp = param.grad.clone() * mask.view(param.grad.size())
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")
        # update error buffers
        if self.error_correction == 'EF':
            self.e[name] += param.grad - comp
        elif self.error_correction == 'EF21':
            self.g[name] += comp
            return self.g[name]
        return comp