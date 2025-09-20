import torch
import torch.nn as nn

def targeted_ifgsm(model, x, t, eps, steps, alpha, early_stop=True):
    if alpha is None:
        alpha = eps / steps

    x0 = x.detach()
    x_adv = x0.clone().detach()
    target = torch.tensor([t], dtype=torch.long, device=x0.device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        # enable grad on the current x_adv
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss_t = loss_fn(logits, target)

        model.zero_grad(set_to_none=True)
        # print(x_adv.grad)
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss_t.backward()

        # eta = alpha · sign(∇_x loss_t)
        eta = alpha * x_adv.grad.sign()

        with torch.no_grad():
            # print("111111111111111111111111111111111111111111")
            x_next = x_adv - eta
            # project to L_inf ball around the original x0
            delta = torch.clamp(x_next - x0, min=-eps, max=eps)
            x_adv = (x0 + delta).detach()
        probs = torch.softmax(model(x_adv), dim=1)
        print("Target prob:", probs[0, t].item())
        
        # early stop
        if early_stop and model(x_adv).argmax(1).item() == t:
            break

    return x_adv
