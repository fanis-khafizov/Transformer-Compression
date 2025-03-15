import torch
from torch.func import functional_call

def mirror_descent(model, X_train, y_train, param_name, impact, lr, eta, lambda_value, num_steps, criterion, start=None, e=None):

    original_param = dict(model.named_parameters())[param_name]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    if e is None:
        param_grad = torch.autograd.grad(loss, original_param)[0]
    else:
        param_grad = torch.autograd.grad(loss, original_param)[0] + e
    
    if start == 'ones':
        with torch.no_grad():
            impact = torch.ones_like(param_grad) / param_grad.numel()
    
    impact = impact.detach().requires_grad_(True)

    new_params = {param_name: original_param.clone()}

    for _ in range(num_steps):
        # Update parameter using impact
        param_new = original_param - lr * impact * param_grad.detach()
        new_params[param_name] = param_new
        # Compute outputs with new parameters
        outputs_new = functional_call(model, new_params, (X_train,))
        # Compute new loss
        loss_new = criterion(outputs_new, y_train)

        # Compute gradient of new loss w.r.t. impact
        grad_impact = torch.autograd.grad(loss_new, impact)[0]

        with torch.no_grad():
            impact_update = torch.pow(impact, 1/(1+eta*lambda_value)).detach() * torch.exp(-(eta/(1+eta*lambda_value)) * (grad_impact))
            impact = impact_update / impact_update.sum()

        # Ensure impact requires grad for the next iteration
        impact.requires_grad_(True)

    return impact.detach()


def gradient_descent(model, X_train, y_train, param_name, impact, lr, eta, num_steps, criterion, start=None, scale=1.0):
    
    original_param = dict(model.named_parameters())[param_name]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    param_grad = torch.autograd.grad(loss, original_param, retain_graph=True, create_graph=True)[0]

    if start == 'topk':
        impact = torch.zeros_like(param_grad).detach()
        topk_indices = torch.topk(param_grad.abs().view(-1), int(0.001 * param_grad.numel())).indices
        impact.view(-1)[topk_indices] = 1.0
    elif start == 'ones':
        impact = torch.ones_like(param_grad)
    elif start == 'center':
        impact = (torch.ones_like(param_grad) / 2)

    impact = impact.clone().detach().requires_grad_(True)
    
    new_params = {name: param.clone() for name, param in model.named_parameters()}

    for _ in range(num_steps):
        # Update parameter using impact
        param_new = original_param - lr * impact * param_grad
        # Create new parameter dictionary
        new_params[param_name] = param_new
        # Compute outputs with new parameters
        outputs_new = functional_call(model, new_params, (X_train,))
        # Compute new loss
        loss_new = criterion(outputs_new, y_train)

        # Compute gradient of new loss w.r.t. impact
        grad_impact = torch.autograd.grad(loss_new, impact)[0]

        with torch.no_grad():
            impact = impact.detach() - eta * lr * grad_impact.detach()
            impact = torch.clip(impact, 0, scale)
        
        # Ensure impact requires grad for the next iteration
        impact.requires_grad_(True)

    return impact.detach()