from math import ceil
from matplotlib import pyplot as plt
import torch
from descent import gradient_descent, mirror_descent

class TopK:
    def __init__(self, k):
        self.k = k
    
    def update(self, *args, **kwargs):
        pass

    def compress(self, name, param):
        k = ceil(self.k * param.numel())
        tensor = param.grad.view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        return compressed_tensor


class TopK_EF21:
    def __init__(self, k, model):
        self.k = k
        self.g = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    
    def update(self, *args, **kwargs):
        pass

    def compress(self, name, param):
        # compression of difference
        k = ceil(self.k * param.numel())
        tensor = (param.grad - self.g[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        # update g
        self.g[name] += compressed_tensor
        return self.g[name]

class TopK_EF:
    def __init__(self, k, model):
        self.k = k
        self.e = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    
    def update(self, *args, **kwargs):
        pass

    def compress(self, name, param):
        # compression of difference
        k = ceil(self.k * param.numel())
        tensor = (param.grad + self.e[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        # update e
        self.e[name] += param.grad - compressed_tensor
        return compressed_tensor

class RandK:
    def __init__(self, k):
        self.k = k
    
    def update(self, *args, **kwargs):
        pass

    def compress(self, name, param):
        k = ceil(self.k * param.numel())
        tensor = param.grad
        mask = torch.randperm(tensor.numel()) < k
        mask = mask.view(tensor.size())
        compressed_tensor = tensor * mask
        return compressed_tensor


class ImpK_b:
    def __init__(self, model, k, start='ones'):
        self.model = model
        self.k = k
        self.w = {name: torch.ones_like(param) / param.numel()
            for name, param in model.named_parameters()
        }
        self.start = start

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = mirror_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                lambda_value=0.001,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start
            )
            # plt.hist(self.w[name].cpu().detach().flatten(), bins=50, label=name)
            # plt.show()
            # print(f'{name} min: {self.w[name].min():.5f}, max: {self.w[name].max():.5f}, min/max: {self.w[name].min()/self.w[name].max():.3f}')

    def compress(self, name, param):
        k = ceil(self.k * param.numel())

        tensor = (param.grad * self.w[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor *= k / (self.w[name].flatten()[topk_indices].sum())
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size

        return compressed_tensor

class ImpK_b_EF21:
    def __init__(self, model, k, start='abs'):
        self.model = model
        self.k = k
        self.g = {name: (imp := torch.zeros_like(param))
            for name, param in model.named_parameters()
        }
        self.w = {name: (imp := torch.ones_like(param))
            for name, param in model.named_parameters()
        }
        self.start = start

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = mirror_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                lambda_value=0.001,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start,
            )

    def compress(self, name, param):
        k = ceil(self.k * param.numel())
        tensor = (param.grad * self.w[name] - self.g[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor *= k / (self.w[name].flatten()[topk_indices].sum())
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        # update g
        self.g[name] += compressed_tensor
        return self.g[name]

class ImpK_b_EF:
    def __init__(self, model, k, start='abs'):
        self.model = model
        self.k = k
        self.e = {name: (imp := torch.zeros_like(param))
            for name, param in model.named_parameters()
        }
        self.w = {name: (imp := torch.ones_like(param)) / param.numel()
            for name, param in model.named_parameters()
        }
        self.start = start

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = mirror_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                lambda_value=0.001,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start,
                e=self.e[name]
            )
            # plt.hist(self.w[name].cpu().detach().flatten(), bins=50, label=name)
            # plt.show()
            # print(f'{name} min: {self.w[name].min():.5f}, max: {self.w[name].max():.5f}, min/max: {self.w[name].min()/self.w[name].max():.3f}')


    def compress(self, name, param):
        k = ceil(self.k * param.numel())
        
        tensor = ((param.grad + self.e[name]) * self.w[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor *= k / (self.w[name].flatten()[topk_indices].sum())
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        # update e
        self.e[name] += param.grad - compressed_tensor

        return compressed_tensor


class ImpK_c:
    def __init__(self, model, k, start='ones', scale=1.0):
        self.model = model
        self.k = k
        self.w = {name: (imp := torch.ones_like(param))
            for name, param in model.named_parameters()
        }
        self.start = start
        self.scale = scale

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = gradient_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start,
                scale=self.scale
            )
            # plt.hist(self.w[name].cpu().detach().flatten(), bins=50, label=name)
            # plt.show()
            # print(f'{name} min: {self.w[name].min():.5f}, max: {self.w[name].max():.5f}, min/max: {self.w[name].min()/self.w[name].max():.3f}')


    def compress(self, name, param):
        k = ceil(self.k * param.numel())

        tensor = (param.grad * self.w[name]).view(-1) # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        
        # Apply mask to tensor
        compressed_tensor = tensor * mask
        compressed_tensor = compressed_tensor.view(param.grad.size())

        return compressed_tensor

class ImpK_c_EF21:
    def __init__(self, model, k, start='ones', scale=1.0):
        self.model = model
        self.k = k
        self.g = {name: (imp := torch.zeros_like(param))
            for name, param in model.named_parameters()
        }
        self.w = {name: (imp := torch.ones_like(param))
            for name, param in model.named_parameters()
        }
        self.start = start
        self.scale = scale

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = gradient_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start,
                scale=self.scale
            )
            # plt.hist(self.w[name].cpu().detach().flatten(), bins=50, label=name)
            # plt.show()
            # print(f'{name} min: {self.w[name].min():.5f}, max: {self.w[name].max():.5f}, min/max: {self.w[name].min()/self.w[name].max():.3f}')


    def compress(self, name, param):
        k = ceil(self.k * param.numel())
        tensor = (param.grad * self.w[name] - self.g[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        # update g
        self.g[name] += compressed_tensor
        return self.g[name]

class ImpK_c_EF:
    def __init__(self, model, k, start='ones', scale=1.0):
        self.model = model
        self.k = k
        self.e = {name: (imp := torch.zeros_like(param))
            for name, param in model.named_parameters()
        }
        self.w = {name: (imp := torch.ones_like(param))
            for name, param in model.named_parameters()
        }
        self.start = start
        self.scale = scale

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = gradient_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start,
                scale=self.scale
            )
            # plt.hist(self.w[name].cpu().detach().flatten(), bins=50, label=name)
            # plt.show()
            # print(f'{name} min: {self.w[name].min():.5f}, max: {self.w[name].max():.5f}, min/max: {self.w[name].min()/self.w[name].max():.3f}')


    def compress(self, name, param):
        k = ceil(self.k * param.numel())
        
        tensor = ((param.grad + self.e[name]) * self.w[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor *= k / (self.w[name].flatten()[topk_indices].sum())
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        # update e
        self.e[name] += param.grad - compressed_tensor

        return compressed_tensor
    
    
    
    

######## SCAM - Stochastic Conditional Accelerated Method

class SCAM_b_EF:
    def __init__(self, model, k, start='abs'):
        self.model = model
        self.k = k
        self.e = {name: (imp := torch.zeros_like(param))
            for name, param in model.named_parameters()
        }
        self.w = {name: (imp := torch.ones_like(param)) / param.numel()
            for name, param in model.named_parameters()
        }
        self.start = start

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = mirror_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                lambda_value=0.001,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start,
                e=self.e[name]
            )
            # plt.hist(self.w[name].cpu().detach().flatten(), bins=50, label=name)
            # plt.show()
            # print(f'{name} min: {self.w[name].min():.5f}, max: {self.w[name].max():.5f}, min/max: {self.w[name].min()/self.w[name].max():.3f}')


    def compress(self, name, param):
        k = ceil(self.k * param.numel())
        
        tensor = ((param.grad + self.e[name]) * self.w[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor)
        mask[topk_indices] = self.w[name].flatten()[topk_indices]
        mask *= k / mask.sum()
        
        param_grad_copy = param.grad.clone()
        compressed_tensor = param_grad_copy * mask.reshape(param.shape)
        compressed_tensor = compressed_tensor.reshape(param.shape)  # Reshape back to original size
        # update e
        self.e[name] += param_grad_copy - compressed_tensor
        return compressed_tensor
    
    

class SCAM_c_EF:
    def __init__(self, model, k, start='ones', scale=1.0):
        self.model = model
        self.k = k
        self.e = {name: (imp := torch.zeros_like(param))
            for name, param in model.named_parameters()
        }
        self.w = {name: (imp := torch.ones_like(param))
            for name, param in model.named_parameters()
        }
        self.start = start
        self.scale = scale

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        for name, param in self.model.named_parameters():
            if 'bn' in name or 'shortcut.1' in name:
                continue
            self.w[name] = gradient_descent(
                model=self.model,
                param_name=name,
                impact=self.w[name],
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                start=self.start,
                scale=self.scale
            )
            # plt.hist(self.w[name].cpu().detach().flatten(), bins=50, label=name)
            # plt.show()
            # print(f'{name} min: {self.w[name].min():.5f}, max: {self.w[name].max():.5f}, min/max: {self.w[name].min()/self.w[name].max():.3f}')


    def compress(self, name, param):
        # k = int(self.k * param.numel())
        # tensor = (param.grad * self.w[name] + self.e[name]).view(-1)  # Flatten the tensor to a vector
        # topk_values, topk_indices = tensor.abs().topk(k)
        # mask = torch.zeros_like(tensor, dtype=torch.bool)
        # mask.scatter_(0, topk_indices, True)
        # compressed_tensor = tensor * mask
        # compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        # # update e
        # self.e[name] += param.grad - compressed_tensor
        # return compressed_tensor
        k = ceil(self.k * param.numel())
        
        tensor = ((param.grad + self.e[name]) * self.w[name]).view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor)
        mask[topk_indices] = self.w[name].flatten()[topk_indices]
        # mask *= k / mask.sum()
        
        param_grad_copy = param.grad.clone()
        compressed_tensor = param_grad_copy * mask.reshape(param.shape)
        compressed_tensor = compressed_tensor.reshape(param.shape)  # Reshape back to original size
        # update e
        self.e[name] += param_grad_copy - compressed_tensor
        return compressed_tensor