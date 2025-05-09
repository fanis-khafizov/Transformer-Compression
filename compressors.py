from math import ceil
import torch
from descent import gradient_descent, mirror_descent

# Generic configurable compressor to select strategy, error correction, and update task
class Compressor:
    def __init__(self, model, k, strategy='TopK', error_correction='none', update_task=None, lr=None, update_kwargs=None):
        self.model = model
        self.k = k
        self.strategy = strategy
        self.error_correction = error_correction
        self.lr = lr
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
        return 'ln' in name or 'bias' in name or 'wte' in name or 'wpe' in name

    def update(self, batch):
        if not self.update_task:
            return
        
        update_fn = {
            'mirror_descent': mirror_descent,
            'gradient_descent': gradient_descent
        }[self.update_task]
        
        full_kwargs = dict(self.update_kwargs)

        if self.error_correction == 'EF':
            full_kwargs['errors'] = self.e
        
        self.w = update_fn(
            self.model,
            batch,
            lr=self.lr,
            **full_kwargs,
        )
        # for name, param in self.model.named_parameters():
        #     if not self.skip(name):
        #         plt.hist(self.w[name].view(-1).cpu().numpy(), bins=100, alpha=0.5, label=name)
        #         plt.title(f"Histogram of {name}")
        #         plt.xlabel("Value")
        #         plt.ylabel("Frequency")
        #         plt.legend()
        #         plt.show()

    def compress(self, name, param):
        with torch.no_grad():
            k = ceil(self.k * param.numel())
            
            grad = param.grad.detach()
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
                comp = param.grad * mask.view(param.grad.size())
            elif self.strategy == 'SCAM_TopK':
                flat = grad.view(-1)
                topk_vals, topk_idx = flat.abs().topk(k)
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask.scatter_(0, topk_idx, True)
                comp = mask.view(param.grad.size()) * param.grad
            else:
                raise ValueError(f"Unknown strategy {self.strategy}")

            # update error buffers without autograd
            if self.error_correction == 'EF':
                self.e[name].add_(param.grad.detach() - comp)
            elif self.error_correction == 'EF21':
                self.g[name].add_(comp)
                return self.g[name]
            return comp