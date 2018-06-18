from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as mm
import numpy as np
import torch

PRINT = True

class CNNBiasAll(nn.Module):
    ''' Adds a bias to all convolution layers '''
    def __init__(self, n_past_lidars, n_past_steps, n_future_steps):
        super(CNNBiasAll, self).__init__()

        # Change these values if you want to change the number of steps in the
        # input or output
        values_width = 11 # the number of values on each row in values
        self.vs = (n_past_steps + 1) * values_width # values size
        self.os = n_future_steps * 2 # output size

        # encoder
        self.conv_e0 = nn.Conv2d(n_past_lidars+1, 4, 3, padding=1)
        self.bias_e0 = nn.Linear(self.vs, 4 ) # 330 x 8

        self.conv_e1 = nn.Conv2d(4, 8, 3, padding=1)
        self.bias_e1 = nn.Linear(self.vs, 8 ) # 330 x 8

        self.conv_e2 = nn.Conv2d(8,12,3, padding=1)
        self.bias_e2 = nn.Linear(self.vs, 12 ) # 330 x 8

        self.conv_e3 = nn.Conv2d(12,16,3, padding=1)
        self.bias_e3 = nn.Linear(self.vs, 16 ) # 330 x 8
        self.maxpool_e3 = nn.MaxPool2d(2, stride=2)

        self.conv_e4 = nn.Conv2d(16,20,3, padding=1)
        self.bias_e4 = nn.Linear(self.vs, 20 ) # 330 x 8

        self.conv_e5 = nn.Conv2d(20,24,3, padding=1)
        self.bias_e5 = nn.Linear(self.vs, 24 ) # 330 x 8

        self.conv_e6 = nn.Conv2d(24,28,3, padding=1)
        self.bias_e6 = nn.Linear(self.vs, 28 ) # 330 x 8

        self.conv_e7 = nn.Conv2d(28,32,3, padding=1)
        self.bias_e7 = nn.Linear(self.vs, 32 ) # 330 x 8
        self.maxpool_e7 = nn.MaxPool2d(2, stride=2)

        self.conv_e8 = nn.Conv2d(32,40,3, padding=1)
        self.bias_e8 = nn.Linear(self.vs, 40 ) # 330 x 8

        self.conv_e9 = nn.Conv2d(40,48,3, padding=1)
        self.bias_e9 = nn.Linear(self.vs, 48 ) # 330 x 8

        self.conv_e10 = nn.Conv2d(48,64,3, padding=1)
        self.bias_e10 = nn.Linear(self.vs, 64 ) # 330 x 8
        self.maxpool_e10 = nn.MaxPool2d(3, stride=3)

        # context modules
        self.spatial_dropout = nn.Dropout2d(p=0.2)

        self.conv_c0 = nn.Conv2d(64,96,3, padding=1, dilation=1)
        self.bias_c0 = nn.Linear(self.vs,96 ) # 330 x 8

        self.conv_c1 = nn.Conv2d(96,96,3, padding=1, dilation=1)
        self.bias_c1 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c2 = nn.Conv2d(96,96,3, padding=(2,1), dilation=(2,1))
        self.bias_c2 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c3 = nn.Conv2d(96,96,3, padding=(4,2), dilation=(4,2))
        self.bias_c3 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c4 = nn.Conv2d(96,96,3, padding=(8,4), dilation=(8,4))
        self.bias_c4 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c5 = nn.Conv2d(96,96,3, padding=(12,8), dilation=(12,8))
        self.bias_c5 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c6 = nn.Conv2d(96,96,3, padding=(16,12), dilation=(16,12))
        self.bias_c6 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c7 = nn.Conv2d(96,96,3, padding=(20,16), dilation=(20,16))
        self.bias_c7 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c8 = nn.Conv2d(96,96,3, padding=(24,20), dilation=(24,20))
        self.bias_c8 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c9 = nn.Conv2d(96,96,3, padding=(28,24), dilation=(28,24))
        self.bias_c9 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c10 = nn.Conv2d(96,96,3, padding=(32,28), dilation=(32,28))
        self.bias_c10 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c11 = nn.Conv2d(96,96,3, padding=(1,32), dilation=(1,32))
        self.bias_c11 = nn.Linear(self.vs, 96 ) # 330 x 8

        self.conv_c12 = nn.Conv2d(96,64,3, padding=1, dilation=1)
        self.bias_c12 = nn.Linear(self.vs, 64 ) # 330 x 8

        self.conv_c13 = nn.Conv2d(64,32,3, padding=1, dilation=1)
        self.bias_c13 = nn.Linear(self.vs, 32 ) # 330 x 8

        self.conv_c14 = nn.Conv2d(32,16,3, padding=1, dilation=1)
        self.bias_c14 = nn.Linear(self.vs, 16 ) # 330 x 8

        # decoder convolutions # 150 x 150 x 16
        self.conv_d0 = nn.Conv2d(16,16,3, padding=1)
        self.bias_d0 = nn.Linear(self.vs, 16 ) # 330 x 8

        self.conv_d1 = nn.Conv2d(16,16,3, padding=1)
        self.bias_d1 = nn.Linear(self.vs, 16 ) # 330 x 8
        self.maxpool_d1 = nn.MaxPool2d(2, stride=2)            # 25 x 25 x 16

        self.conv_d2 = nn.Conv2d(16,16,3, padding=1)
        self.bias_d2 = nn.Linear(self.vs, 16 ) # 330 x 8

        self.conv_d3 = nn.Conv2d(16,16,3, padding=1)
        self.bias_d3 = nn.Linear(self.vs, 16 ) # 330 x 8

        self.conv_d4 = nn.Conv2d(16,16,3, padding=1)
        self.bias_d4 = nn.Linear(self.vs, 16 ) # 330 x 8

        # fully connected decoder layers to output
        #self.linear_d2 = nn.Linear(10330,800)
        self.linear_d0 = nn.Linear(25*25*16 + self.vs, 800)
        self.linear_d1 = nn.Linear(800, 512)
        self.linear_d2 = nn.Linear(512, 256)
        self.linear_d3 = nn.Linear(256, 128)
        self.linear_d4 = nn.Linear(128, self.os)

        # Input l: 600 x 600 x 1, v: 30 x 11
    def forward(self, l, v):

        # input
        v = v.view(-1, self.vs)                                   # [2, 330]

        # encoder
        b = F.elu(self.bias_e0(v))                              # [2, 8]
        b = expand_biases(b,600,600)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e0(l), 1, b))             # [2, 8, 600, 600]

        b = F.elu(self.bias_e1(v))                              # [2, 8]
        b = expand_biases(b,600,600)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e1(l), 1, b))             # [2, 8, 600, 600]

        b = F.elu(self.bias_e2(v))                              # [2, 16]
        b = expand_biases(b,600,600)                            # [2, 16, 300, 300]
        l = F.elu(torch.add(self.conv_e2(l), 1, b))             # [2, 16, 300, 300]

        b = F.elu(self.bias_e3(v))                              # [2, 16]
        b = expand_biases(b,600,600)                            # [2, 16, 300, 300]
        l = F.elu(torch.add(self.conv_e3(l), 1, b))             # [2, 16, 300, 300]
        l = self.maxpool_e3(l)                                  # [2, 16, 150, 150]

        b = F.elu(self.bias_e4(v))                              # [2, 8]
        b = expand_biases(b,300,300)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e4(l), 1, b))             # [2, 8, 600, 600]

        b = F.elu(self.bias_e5(v))                              # [2, 8]
        b = expand_biases(b,300,300)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e5(l), 1, b))             # [2, 8, 600, 600]

        b = F.elu(self.bias_e6(v))                              # [2, 8]
        b = expand_biases(b,300,300)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e6(l), 1, b))             # [2, 8, 600, 600]

        b = F.elu(self.bias_e7(v))                              # [2, 8]
        b = expand_biases(b,300,300)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e7(l), 1, b))             # [2, 8, 600, 600]
        l = self.maxpool_e7(l)                                  # [2, 16, 150, 150]

        b = F.elu(self.bias_e8(v))                              # [2, 8]
        b = expand_biases(b,150,150)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e8(l), 1, b))             # [2, 8, 600, 600]

        b = F.elu(self.bias_e9(v))                              # [2, 8]
        b = expand_biases(b,150,150)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e9(l), 1, b))             # [2, 8, 600, 600]

        b = F.elu(self.bias_e10(v))                             # [2, 8]
        b = expand_biases(b,150,150)                            # [2, 8, 600, 600]
        l = F.elu(torch.add(self.conv_e10(l), 1, b))            # [2, 8, 600, 600]
        l = self.maxpool_e10(l)                                  # [2, 16, 50, 50]

        #context module
        b = F.elu(self.bias_c0(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c0(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c1(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c1(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c2(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c2(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c3(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c3(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c4(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c4(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c5(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c5(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c6(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c6(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c7(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c7(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c8(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c8(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c9(v))                              # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c9(l), 1, b)                    # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c10(v))                             # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c10(l), 1, b)                   # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]

        b = F.elu(self.bias_c11(v))                             # [2, 96]
        b = expand_biases(b,50,50)                            # [2, 96, 150, 150]
        l = torch.add(self.conv_c11(l), 1, b)                   # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 96, 150, 150]s

        b = F.elu(self.bias_c12(v))                             # [2, 16]
        b = expand_biases(b,50,50)                            # [2, 16, 150, 150]
        l = torch.add(self.conv_c12(l), 1, b)                   # [2, 16, 150, 150]
        l = F.elu(l)                                            # [2, 16, 150, 150]

        b = F.elu(self.bias_c13(v))                             # [2, 16]
        b = expand_biases(b,50,50)                            # [2, 16, 150, 150]
        l = torch.add(self.conv_c13(l), 1, b)                   # [2, 16, 150, 150]
        l = F.elu(l)                                            # [2, 16, 150, 150]

        b = F.elu(self.bias_c14(v))                             # [2, 16]
        b = expand_biases(b,50,50)                            # [2, 16, 150, 150]
        l = torch.add(self.conv_c14(l), 1, b)                   # [2, 16, 150, 150]
        l = F.elu(l)                                            # [2, 16, 150, 150]

        # decoder convolutions
        b = F.elu(self.bias_d0(v))                              # [2, 16]
        b = expand_biases(b,50,50)                            # [2, 16, 150, 150]
        l = F.elu(torch.add(self.conv_d0(l), 1, b))             # [2, 16, 150, 150]

        b = F.elu(self.bias_d1(v))                              # [2, 16]
        b = expand_biases(b,50,50)                              # [2, 16, 50, 50]
        l = F.elu(torch.add(self.conv_d1(l), 1, b))             # [2, 16, 50, 50]
        l = self.maxpool_d1(l)                                  # [2, 16, 25, 25]

        b = F.elu(self.bias_d2(v))                              # [2, 16]
        b = expand_biases(b,25,25)                              # [2, 16, 50, 50]
        l = F.elu(torch.add(self.conv_d2(l), 1, b))             # [2, 16, 50, 50]

        b = F.elu(self.bias_d3(v))                              # [2, 16]
        b = expand_biases(b,25,25)                              # [2, 16, 50, 50]
        l = F.elu(torch.add(self.conv_d3(l), 1, b))             # [2, 16, 50, 50]

        b = F.elu(self.bias_d4(v))                              # [2, 16]
        b = expand_biases(b,25,25)                              # [2, 16, 50, 50]
        l = F.elu(torch.add(self.conv_d4(l), 1, b))             # [2, 16, 50, 50]

        # fully connected decoder layers to output
        l = l.view(-1, 25*25*16)                            # 10000
        y = torch.cat((l,v),1)                              # 10330
        y = F.elu(self.linear_d0(y))                          # 800
        y = F.elu(self.linear_d1(y))                          # 300
        y = F.elu(self.linear_d2(y))                          # 300
        y = F.elu(self.linear_d3(y))                          # 300
        y = self.linear_d4(y)                                 # 60

        return y

def expand_biases(v, w, h):
    b = v.size(0) # batch size
    c = v.size(1) # number of channels
    return v.unsqueeze(2).expand(b,c,h).unsqueeze(3).expand(b,c,h,w)

def print_size(tensor, cond):
    if cond:
        print(tensor.size())
