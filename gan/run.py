from torch import nn
from torch.autograd.grad_mode import F
import torch

m = nn.ConvTranspose2d(100,512,4,1,0,bias=False)
input = torch.randn(100)
output = m(input)
print(output)