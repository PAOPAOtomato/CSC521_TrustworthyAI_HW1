import torch
import torch.nn as nn
from tartged_IFGSM import targeted_ifgsm

# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 1
eps = 0.62
steps = 20
alpha = eps / steps

adv_x = targeted_ifgsm(N, x, t, eps=eps, steps=steps, alpha=alpha)

print("Orig → New:", N(x).argmax(1).item(), "→", N(adv_x).argmax(1).item())
print("Norm of difference:", torch.norm((adv_x - x).flatten(), p=float('inf')).item())