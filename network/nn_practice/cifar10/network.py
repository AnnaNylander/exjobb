from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as mm
import numpy

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3,128,2,padding=1) #with relu
        self.conv2 = nn.Conv2d(128,64,2,padding=1, dilation=2)
        self.conv2_1 = nn.Conv2d(64,64,2,padding=1, dilation=4)
        self.norm1 = mm.LocalResponseNorm(4)#,alpha=0.001/9.0) #beta and k (bias) is default
        self.conv3 = nn.Conv2d(64,64,3,padding=1, stride=2)
        self.maxpool1 = nn.MaxPool2d(2, padding=1)
        self.norm2 = mm.LocalResponseNorm(4)#,alpha=0.001/9.0) #beta and k (bias) is default

        self.lin1 = nn.Linear(9*9*64, 2500)
        self.lin3 = nn.Linear(2500, 512)
        self.lin4 = nn.Linear(512, 192)

        self.lin5 = nn.Linear(192, 10)



    def forward(self, x):# 32 x 32 x 3
        x = self.conv1(x) # 32 x 32 x 128
        x = F.relu(x)
        x = self.conv2(x) # 33 x 33 x 64
        x = F.relu(x)
        x = self.conv2_1(x) # 31 x 31 x 64
        x = self.norm1(x)
        x = self.conv3(x) # 16 x 16 x 64
        x = self.maxpool1(x) # 9 x 9 x 64
        x = self.norm2(x)
        x = x.view(-1, 9*9*64 )

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        x = F.relu(x)

        x = self.lin5(x) #10
        return x

class Tf_tutorial_Network(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()

        self.conv1 = nn.Conv2d(3,64,5,padding=2) #with relu
        self.maxpool1 = nn.MaxPool2d(3,stride=2,padding=1)
        self.norm1 = mm.LocalResponseNorm(4)#,alpha=0.001/9.0) #beta and k (bias) is default
        self.conv2 = nn.Conv2d(64,64,5, padding=2) #with relu
        self.norm2 = mm.LocalResponseNorm(4)#,alpha=0.001/9.0)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.lin1 = nn.Linear(8*8*64,384)
        self.lin2 = nn.Linear(384, 192)

        self.lin3 = nn.Linear(192, 10)



    def forward(self, x):# 32 x 32 x 3
        x = self.conv1(x) # 32 x 32 x 64
        x = F.relu(x)
        x = self.maxpool1(x) # 16 x 16 x 64
        x = self.norm1(x) # 16 x 16 x 64
        x = self.conv2(x) # 16 x 16 x 64
        x = F.relu(x)
        x = self.norm2(x) # 16 x 16 x 64
        x = self.maxpool2(x) # 8 x 8 x 64

        x = x.view(-1, 8*8*64) # 16384
        x = self.lin1(x) #384
        x = F.relu(x)
        x = self.lin2(x) #192
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
