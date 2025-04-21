import torch
from torch.func import functional_call

def mirror_descent(model, batch, param_name, impact, lr, eta, lambda_value, num_steps, start=None, error=None):

    original_param = dict(model.named_parameters())[param_name]

    outputs = model(batch, labels=batch)
    loss = outputs.loss
    
    if error is None:
        param_grad = torch.autograd.grad(loss, original_param)[0]
    else:
        param_grad = torch.autograd.grad(loss, original_param)[0] + error
    
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
        outputs_new = functional_call(model, new_params, (batch,), {'labels': batch})
        # Compute new loss
        loss_new = outputs_new.loss

        # Compute gradient of new loss w.r.t. impact
        grad_impact = torch.autograd.grad(loss_new, impact)[0]

        with torch.no_grad():
            impact_update = torch.pow(impact, 1/(1+eta*lambda_value)).detach() * torch.exp(-(eta/(1+eta*lambda_value)) * (grad_impact))
            impact = impact_update / impact_update.sum()

        # Ensure impact requires grad for the next iteration
        impact.requires_grad_(True)

    return impact.detach()


def gradient_descent(model, batch, param_name, impact, lr, eta, num_steps, start=None, scale=1.0, error=None):
    
    original_param = dict(model.named_parameters())[param_name]

    outputs = model(batch, labels=batch)
    loss = outputs.loss
    
    if error is None:
        param_grad = torch.autograd.grad(loss, original_param)[0]
    else:
        param_grad = torch.autograd.grad(loss, original_param)[0] + error
    
    if start == 'ones':
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
        outputs_new = functional_call(model, new_params, (batch,), {'labels': batch})
        # Compute new loss
        loss_new = outputs_new.loss

        # Compute gradient of new loss w.r.t. impact
        grad_impact = torch.autograd.grad(loss_new, impact)[0]

        with torch.no_grad():
            impact = impact.detach() - eta * lr * grad_impact.detach()
            impact = torch.clip(impact, 0, scale)
        
        # Ensure impact requires grad for the next iteration
        impact.requires_grad_(True)

    return impact.detach()

def mirror_descent_full(model, batch, impacts, lr, eta, lambda_value, num_steps, start=None, errors=None):
    # 1) считаем один раз начальный градиент по весам
    outputs = model(batch, labels=batch)
    loss = outputs.loss
    base_grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()])
    if errors is not None:
        base_grads = [g + errors[name] for g, (name, _) in zip(base_grads, model.named_parameters())]

    # 2) инициализируем impacts
    skip = lambda n: 'ln' in n
    for name, param in model.named_parameters():
        if start == 'ones':
            imp = torch.ones_like(param)
        elif start == 'center':
            imp = torch.ones_like(param).div_(2)
        else:
            imp = impacts[name]
        impacts[name] = imp.detach().requires_grad_(True)

    # 3) подготовим «новые» параметры на каждый шаг
    new_params = {n: p.clone() for n, p in model.named_parameters()}

    # список тех impact‑тензоров, для которых мы считаем градиент
    impact_keys = [n for n in impacts if not skip(n)]
    for _ in range(num_steps):
        # a) обновляем new_params по старым весам и base_grads
        for (name, orig_p), g in zip(model.named_parameters(), base_grads):
            if skip(name): continue
            new_params[name] = orig_p - lr * impacts[name] * g

        # b) прямой проход с new_params
        out_new = functional_call(model, new_params, (batch,), {'labels': batch})
        loss_new = out_new.loss

        # c) сразу получаем все градиенты impact‑ов
        grads_impacts = torch.autograd.grad(
            loss_new,
            [impacts[k] for k in impact_keys],
            allow_unused=True
        )

        # d) обновляем impacts «вдоль градиента»
        with torch.no_grad():
            for k, g_imp in zip(impact_keys, grads_impacts):
                imp = impacts[k]
                impact_update = torch.pow(imp, 1/(1+eta*lambda_value)).detach() * torch.exp(-(eta/(1+eta*lambda_value)) * (g_imp if g_imp is not None else 0))
                impacts[k] = (impact_update / impact_update.sum() * imp.numel()).detach().requires_grad_(True) 

    # 4) финально отключаем grad, если нужно
    for k in impact_keys:
        impacts[k] = impacts[k].detach().requires_grad_(False)

    return impacts

def gradient_descent_full(model, batch, impacts, lr, eta, num_steps, start=None, scale=1.0, errors=None):
    # 1) считаем один раз начальный градиент по весам
    outputs = model(batch, labels=batch)
    loss = outputs.loss
    base_grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()])
    if errors is not None:
        base_grads = [g + errors[name] for g, (name, _) in zip(base_grads, model.named_parameters())]

    # 2) инициализируем impacts
    skip = lambda n: 'ln' in n
    for name, param in model.named_parameters():
        if start == 'ones':
            imp = torch.ones_like(param)
        elif start == 'center':
            imp = torch.ones_like(param).div_(2)
        else:
            imp = impacts[name]
        impacts[name] = imp.detach().requires_grad_(True)

    # 3) подготовим «новые» параметры на каждый шаг
    new_params = {n: p.clone() for n, p in model.named_parameters()}

    # список тех impact‑тензоров, для которых мы считаем градиент
    impact_keys = [n for n in impacts if not skip(n)]
    for _ in range(num_steps):
        # a) обновляем new_params по старым весам и base_grads
        for (name, orig_p), g in zip(model.named_parameters(), base_grads):
            if skip(name): continue
            new_params[name] = orig_p - lr * impacts[name] * g

        # b) прямой проход с new_params
        out_new = functional_call(model, new_params, (batch,), {'labels': batch})
        loss_new = out_new.loss

        # c) сразу получаем все градиенты impact‑ов
        grads_impacts = torch.autograd.grad(
            loss_new,
            [impacts[k] for k in impact_keys],
            allow_unused=True
        )

        # d) обновляем impacts «вдоль градиента»
        with torch.no_grad():
            for k, g_imp in zip(impact_keys, grads_impacts):
                imp = impacts[k]
                upd = imp - eta * lr * (g_imp if g_imp is not None else 0)
                impacts[k] = upd.clamp(0, scale).detach().requires_grad_(True)

    # 4) финально отключаем grad, если нужно
    for k in impact_keys:
        impacts[k] = impacts[k].detach().requires_grad_(False)

    return impacts