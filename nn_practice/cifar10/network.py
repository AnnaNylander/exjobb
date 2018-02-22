from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, side, H, D_out,kernel_size):
        super(Network, self).__init__()

        self.kernel_size = kernel_size
        self.unkernel = side-kernel_size + 1

        self.conv1 = nn.Conv2d(3,5,kernel_size)
        self.lin1 = nn.Linear(5*self.unkernel*self.unkernel,H)
        self.lin2 = nn.Linear(H, D_out)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 5*self.unkernel*self.unkernel)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
