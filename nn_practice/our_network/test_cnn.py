# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    #in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
    nn.Conv3d(8, 8, 3, stride=1),
    torch.nn.ELU(),
    nn.Conv3d(8, 8, 3, stride=1),
    torch.nn.ELU(),
    nn.MaxPool3D(2, stride=2),
    torch.nn.ELU(),
    nn.Conv3d(16, 16, 3, stride=1),
    torch.nn.ELU(),
    nn.Conv3d(16, 16, 3, stride=1),
    torch.nn.ELU(),
    nn.Conv3d(16, 16, 3, stride=1),
    torch.nn.ELU(),
    nn.MaxPool3D(2, stride=2),
    torch.nn.ELU(),
    # Context module
    torch.nn.Linear(H, D_out),
)
