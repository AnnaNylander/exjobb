from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as mm
import numpy
import torch

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        #encoder
        self.conv0 = nn.Conv2d(1,8,3, padding=1) # with elu #twice!
        self.conv1 = nn.Conv2d(8,8,3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8,16,3, padding=1) # with elu
        self.conv3 = nn.Conv2d(16,16,3, padding=1) # with elu #twice!
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        # context modules
        # all followed by elu
        # layer 1-12 is also followed by spatial dropout layer with pd=0.20
        self.spatial_dropout = nn.Dropout2d(p=0.2)
        self.Layer1 = nn.Conv2d(16,96,3, padding=1, dilation=1)
        self.Layer2 = nn.Conv2d(96,96,3, padding=1, dilation=1)
        self.Layer3 = nn.Conv2d(96,96,3, padding=(2,1), dilation=(2,1))
        self.Layer4 = nn.Conv2d(96,96,3, padding=(4,2), dilation=(4,2))
        self.Layer5 = nn.Conv2d(96,96,3, padding=(8,4), dilation=(8,4))
        self.Layer6 = nn.Conv2d(96,96,3, padding=(12,8), dilation=(12,8))
        self.Layer7 = nn.Conv2d(96,96,3, padding=(16,12), dilation=(16,12))
        self.Layer8 = nn.Conv2d(96,96,3, padding=(20,16), dilation=(20,16))
        self.Layer9 = nn.Conv2d(96,96,3, padding=(24,20), dilation=(24,20))
        self.Layer10 = nn.Conv2d(96,96,3, padding=(28,24), dilation=(28,24))
        self.Layer11 = nn.Conv2d(96,96,3, padding=(32,28), dilation=(32,28))
        self.Layer12 = nn.Conv2d(96,96,3, padding=(1,32), dilation=(1,32))
        self.Layer13 = nn.Conv2d(96,16,3, padding=1, dilation=1)

        #decoder #all followed by elu
        #concatenate new values here
        self.Linear1 = nn.Linear(360330,1000) # add more data here
        self.Linear2 = nn.Linear(1000,500)
        self.Linear3 = nn.Linear(500, 500)
        self.Linear4 = nn.Linear(500, 60)

    def forward(self, l, v): # 600 x 600 x 1
        # encoder
        l = F.elu(self.conv0(l)) # 600 x 600 x 8
        l = F.elu(self.conv1(l)) # 600 x 600 x 8
        l = self.maxpool1(l) # 300 x 300 x 8
        l = F.elu(self.conv2(l)) # 300 x 300 x 16
        l = F.elu(self.conv3(l)) # 300 x 300 x 16
        l = F.elu(self.conv3(l)) # 300 x 300 x 16
        l = self.maxpool2(l) # 150 x 150 x 16
        #context module
        l = self.spatial_dropout(F.elu(self.Layer1(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer2(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer3(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer4(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer5(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer6(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer7(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer8(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer9(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer10(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer11(l))) # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.Layer12(l))) # 150 x 150 x 96
        l = F.elu(self.Layer13(l)) # 150 x 150 x 16

        # decoder
        l = l.view(-1, 150*150*16) # 360000
        v = v.view(-1, 30*11)
        x = torch.cat((l,v),1) # 360330
        x = F.elu(self.Linear1(x)) # 5000
        x = F.elu(self.Linear2(x)) # 1000
        x = F.elu(self.Linear3(x)) # 500
        x = self.Linear4(x)

        return x
