from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()

        self.conv1 = nn.Conv2d(3,10,3)
        self.conv2 = nn.Conv2d(10,5,5)
        self.lin1 = nn.Linear(26*26*5,1000)
        self.lin2 = nn.Linear(1000, 100)
        self.lin3 = nn.Linear(100,10)


    def forward(self, x):# 32 x 32 x 3
        x = self.conv1(x) # 30 x 30 x 10
        x = F.relu(x)
        x = self.conv2(x) # 26 x 26 x 5
        x = F.relu(x)
        x = x.view(-1, 26*26*5) # 3380
        x = self.lin1(x) #1000
        x = F.relu(x)
        x = self.lin2(x) #100
        x = F.relu(x)
        x = self.lin3(x) #10
        return x

class First_Network(nn.Module):
    def __init__(self, D_out):
        super(First_Network, self).__init__()
        self.side = 32
        self.kernel_size = 5
        self.unkernel = self.side-self.kernel_size + 1
        H = 100

        self.conv1 = nn.Conv2d(3,5,self.kernel_size)
        self.lin1 = nn.Linear(5*self.unkernel*self.unkernel,H)
        self.lin2 = nn.Linear(H, D_out)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 5*self.unkernel*self.unkernel)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
