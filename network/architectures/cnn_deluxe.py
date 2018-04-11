from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as mm
import numpy
import torch

''' Mainly using convolutions throughout the network to reduce the number of
paramters.
'''

class CNNOnly(nn.Module):
    ''' CNN all the way, no fully connected layers '''
    def __init__(self):
        super(CNNOnly, self).__init__()
        # value encoder
        self.conv_v0 = nn.Conv1d(11,30,3, padding=1)                # 11 x 30 x 1
        self.conv_v1 = nn.Conv1d(30,30,3, padding=2, dilation=2)    # 30 x 30 x 1
        self.conv_v2 = nn.Conv1d(30,30,3, padding=4, dilation=4)    # 30 x 30 x 1
        self.conv_v3 = nn.Conv1d(30,30,3, padding=8, dilation=8)    # 30 x 30 x 1

        self.conv_v4 = nn.Conv1d(30,30,3, padding=16, dilation=16)    # 30 x 30 x 1
        self.linear_v4 = nn.Linear(30*30,1) # 1 x 1

        # lidar encoder
        self.conv_e0 = nn.Conv2d(1, 8, 3, padding=1)

        self.conv_e1 = nn.Conv2d(8, 8, 3, padding=1)
        self.maxpool_e1 = nn.MaxPool2d(2, stride=2)

        self.conv_e2 = nn.Conv2d(8, 16, 3, padding=1)

        self.conv_e3 = nn.Conv2d(16, 16, 3, padding=1)
        self.maxpool_e3 = nn.MaxPool2d(2, stride=2)

        self.conv_e4 = nn.Conv2d(16, 30, 3, padding=1)

        # context modules
        self.spatial_dropout = nn.Dropout2d(p=0.2)

        self.conv_c0 = nn.Conv2d(30,96,3, padding=1, dilation=1)
        self.conv_c1 = nn.Conv2d(96,96,3, padding=1, dilation=1)
        self.conv_c2 = nn.Conv2d(96,96,3, padding=(2,1), dilation=(2,1))
        self.conv_c3 = nn.Conv2d(96,96,3, padding=(4,2), dilation=(4,2))
        self.conv_c4 = nn.Conv2d(96,96,3, padding=(8,4), dilation=(8,4))
        self.conv_c5 = nn.Conv2d(96,96,3, padding=(12,8), dilation=(12,8))
        self.conv_c6 = nn.Conv2d(96,96,3, padding=(16,12), dilation=(16,12))
        self.conv_c7 = nn.Conv2d(96,96,3, padding=(20,16), dilation=(20,16))
        self.conv_c8 = nn.Conv2d(96,96,3, padding=(24,20), dilation=(24,20))
        self.conv_c9 = nn.Conv2d(96,96,3, padding=(28,24), dilation=(28,24))
        self.conv_c10 = nn.Conv2d(96,96,3, padding=(32,28), dilation=(32,28))
        self.conv_c11 = nn.Conv2d(96,96,3, padding=(1,32), dilation=(1,32))
        self.conv_c12 = nn.Conv2d(96,30,3, padding=1, dilation=1)

        # decoder convolutions # 150 x 150 x 16
        self.conv_d0 = nn.Conv2d(30,30,3, padding=1)
        self.maxpool_d0 = nn.MaxPool2d(3, stride=3)            # 150 x 150 x 30

        self.conv_d1 = nn.Conv2d(30,30,3, padding=1)
        self.maxpool_d1 = nn.MaxPool2d(3, stride=3)            # 50 x 50 x 30

        self.conv_d2 = nn.Conv2d(30,30,3, padding=1)
        self.maxpool_d2 = nn.MaxPool2d(2, stride=2)            # 25 x 25 x 30

        self.conv_d3 = nn.Conv2d(30,30,3, padding=1)
        self.maxpool_d3 = nn.MaxPool2d(5, stride=5)            # 5 x 5 x 30

        self.conv_d4 = nn.Conv2d(30,30,3, padding=1)
        #self.maxpool_d4 = nn.MaxPool2d(3, stride=2)            # 2 x 1 x 30


        # Input l: 600 x 600 x 1, v: 30 x 11
    def forward(self, l, v):
        # input
        # TODO ensure v is 30x11x1 here
        v = self.spatial_dropout(F.elu(self.conv_v0(v)))
        v = self.spatial_dropout(F.elu(self.conv_v1(v)))
        v = self.spatial_dropout(F.elu(self.conv_v2(v)))
        v = self.spatial_dropout(F.elu(self.conv_v3(v)))

        v = F.elu(self.conv_v4(v))
        v = v.view(-1, 30*30*1)                            # 900
        v = F.elu(linear_v4(v))

        # encoder
        l = F.elu(self.conv_e0(l))

        l = F.elu(self.conv_e1(l))           # 600 x 600 x 8
        l = self.maxpool_e1(l)                                # 300 x 300 x 8

        l = F.elu(self.conv_e2(l))           # 600 x 600 x 8

        l = F.elu(self.conv_e3(l))           # 600 x 600 x 8
        l = self.maxpool_e3(l)                                # 150 x 150 x 16

        l = torch.add(self.conv_e4(l), 1, v)
        l = self.spatial_dropout(F.elu(l))                  # 150 x 150 x 30

        #context module
        l = self.spatial_dropout(F.elu(self.conv_c0(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c1(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c2(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c3(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c4(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c5(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c6(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c7(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c8(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c9(l)))        # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c10(l)))       # 150 x 150 x 96
        l = self.spatial_dropout(F.elu(self.conv_c11(l)))       # 150 x 150 x 96
        l = F.elu(self.conv_c11(l))                             # 150 x 150 x 96

        # decoder convolutions
        l = F.elu(self.conv_d0(l))
        l = self.maxpool_d0(l)                              # 50 x 50 x 16

        l = F.elu(self.conv_d1(l))
        l = self.maxpool_d1(l)                              # 50 x 50 x 16

        # fully connected decoder layers to output
        l = l.view(-1, 25*25*16)                            # 10000
        y = torch.cat((l,v),1)                              # 10330
        y = F.elu(self.linear_d1(y))                          # 800
        y = F.elu(self.linear_d2(y))                          # 300
        y = self.linear_3(y)                                 # 60

        return y

class CNNLSTM(nn.Module):
    ''' Values are processed in an LSTM and then added as a bias '''
    def __init__(self):
        super(CNNOnly, self).__init__()

    def forward(self, l, v):
        return l
