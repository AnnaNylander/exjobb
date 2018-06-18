from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as mm
import numpy
import torch

''' Mainly using convolutions throughout the network to reduce the number of
paramters.
'''
PRINT = True

class CNNOnly(nn.Module):
    ''' CNN all the way, no fully connected layers '''
    def __init__(self, n_past_lidars, n_past_steps, n_future_steps):
        super(CNNOnly, self).__init__()

        # Change these values if you want to change the number of steps in the
        # input or output
        values_width = 11 # the number of values on each row in values
        self.vs = (n_past_steps + 1) * values_width # values size
        self.os = n_future_steps * 2 # output size
        ns = (n_past_steps + 1) # number of past steps including current

        # value encoder
        self.conv_v0 = nn.Conv1d(values_width, ns, 3, padding=1)
        self.conv_v1 = nn.Conv1d(ns, ns,3, padding=2, dilation=2)
        self.conv_v2 = nn.Conv1d(ns, ns,3, padding=4, dilation=4)
        self.conv_v3 = nn.Conv1d(ns, ns,3, padding=8, dilation=8)

        self.conv_v4 = nn.Conv1d(ns,ns,3, padding=16, dilation=16)
        self.maxpool_v4 = nn.MaxPool2d((1,2), stride=(1,2))

        self.conv_v5 = nn.Conv1d(ns,ns,3, padding=1, dilation=1)
        self.maxpool_v5 = nn.MaxPool2d((1,3), stride=(1,3))

        self.conv_v6 = nn.Conv1d(ns,ns,3, padding=1, dilation=1)
        self.maxpool_v6 = nn.MaxPool2d((1,2), stride=(1,1))

        self.conv_v7 = nn.Conv1d(ns,ns,3, padding=1, dilation=1)
        self.maxpool_v7 = nn.MaxPool2d((1,2), stride=(1,2))

        self.conv_v8 = nn.Conv1d(ns,ns,3, padding=1, dilation=1)
        self.maxpool_v8 = nn.MaxPool2d((1,2), stride=(1,2))

        # lidar encoder
        self.conv_e0 = nn.Conv2d(n_past_lidars + 1, 8, 3, padding=1)

        self.conv_e1 = nn.Conv2d(8, 8, 3, padding=1)
        self.maxpool_e1 = nn.MaxPool2d(2, stride=2)

        self.conv_e2 = nn.Conv2d(8, 16, 3, padding=1)

        self.conv_e3 = nn.Conv2d(16, 16, 3, padding=1)
        self.maxpool_e3 = nn.MaxPool2d(2, stride=2)

        self.conv_e4 = nn.Conv2d(16, ns, 3, padding=1)

        # context modules
        self.spatial_dropout = nn.Dropout2d(p=0.2)

        self.conv_c0 = nn.Conv2d(ns,96,3, padding=1, dilation=1)
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
        self.conv_c12 = nn.Conv2d(96,ns,3, padding=1, dilation=1)

        # decoder convolutions # 150 x 150 x 16
        self.conv_d0 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d0 = nn.MaxPool2d(3, stride=3)

        self.conv_d1 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d1 = nn.MaxPool2d(3, stride=3)

        self.conv_d2 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d2 = nn.MaxPool2d(2, stride=2)

        self.conv_d3 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d3 = nn.MaxPool2d(2, stride=2)

        self.conv_d4 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d4 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv_d5 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d5 = nn.MaxPool2d((1,2), stride=(1,2))


        # Input l: 600 x 600 x 1, v: 30 x 11
    def forward(self, l, v):
        # input
        v = v.transpose(1,2) # ensure v is 2 x values_width x ns here         # [2, 11, 30]
        v = self.spatial_dropout(F.elu(self.conv_v0(v)))
        v = self.spatial_dropout(F.elu(self.conv_v1(v)))
        v = self.spatial_dropout(F.elu(self.conv_v2(v)))
        v = self.spatial_dropout(F.elu(self.conv_v3(v)))

        v = F.elu(self.conv_v4(v))
        v = self.maxpool_v4(v)                                  # [2, 30, 15]

        v = F.elu(self.conv_v5(v))                              # [2, 30, 15]
        v = self.maxpool_v5(v)                                  # [2, 30, 5]

        v = F.elu(self.conv_v6(v))                              # [2, 30, 5]
        v = self.maxpool_v6(v)                                  # [2, 30, 4]

        v = F.elu(self.conv_v7(v))                              # [2, 30, 4]
        v = self.maxpool_v7(v)                                  # [2, 30, 2]

        v = F.elu(self.conv_v8(v))                              # [2, 30, 2]
        v = self.maxpool_v8(v)                                  # [2, 30, 1]
        v = v.squeeze()                                         # [2, 30]

        # encoder
        l = F.elu(self.conv_e0(l))                              # [2, 8, 600, 600]

        l = F.elu(self.conv_e1(l))                              # [2, 8, 600, 600]
        l = self.maxpool_e1(l)                                  # [2, 8, 300, 300]
        l = F.elu(self.conv_e2(l))                              # [2, 16, 300, 300]
        l = F.elu(self.conv_e3(l))                              # [2, 16, 300, 300]
        l = self.maxpool_e3(l)                                  # [2, 16, 150, 150]

        v = expand_biases(v, 150, 150)                          # [2, 16, 150, 150]
        l = self.conv_e4(l)                                     # [2, 30, 150, 150]
        l = torch.add(l, 1, v)                                  # [2, 30, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 30, 150, 150]


        #context module
        l = self.spatial_dropout(F.elu(self.conv_c0(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c1(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c2(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c3(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c4(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c5(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c6(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c7(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c8(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c9(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c10(l)))       # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c11(l)))       # [2, 96, 150, 150]
        l = F.elu(self.conv_c12(l))                             # [2, 30, 150, 150]


        # decoder convolutions
        l = F.elu(self.conv_d0(l))                              # [2, 30, 150, 150]
        l = self.maxpool_d0(l)                                  # [2, 30, 50, 50]
        l = F.elu(self.conv_d1(l))                              # [2, 30, 50, 50]
        l = self.maxpool_d1(l)                                  # [2, 30, 16, 16]
        l = F.elu(self.conv_d2(l))                              # [2, 30, 16, 16]
        l = self.maxpool_d2(l)                                  # [2, 30, 8, 8]
        l = F.elu(self.conv_d3(l))                              # [2, 30, 8, 8]
        l = self.maxpool_d3(l)                                  # [2, 30, 4, 4]
        l = F.elu(self.conv_d4(l))                              # [2, 30, 4, 4]
        l = self.maxpool_d4(l)                                  # [2, 30, 2, 2]
        l = F.elu(self.conv_d5(l))                              # [2, 30, 4, 2]
        l = self.maxpool_d5(l)                                  # [2, 30, 2, 1]
        l = l.squeeze()                                         # [2, 30, 2]
        l = l.view(-1,self.os)                                  # [2, self.os]

        return l

class CNNLSTM(nn.Module):
    ''' Values are processed in an LSTM and then added as a bias '''
    def __init__(self, n_past_lidars, n_past_steps, n_future_steps):
        super(CNNLSTM, self).__init__()

        # Change these values if you want to change the number of steps in the
        # input or output
        self.vs = (n_past_steps + 1) * values_width # values size
        self.os = n_future_steps * 2 # output size
        ns = (n_past_steps + 1) # number of past steps including current

        # value encoder LSTM from values to bias
        input_size = 11 # The number of expected features in the input x
        hidden_size = 64 # The number of features in the hidden state h
        num_layers = 2 # Number of recurrent layers
        dropout = 0 # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        bidirectional = False # If True, becomes a bidirectional LSTM

        # There are 30 steps in the sequence, each is a 11x1 vector
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # lidar encoder
        self.conv_e0 = nn.Conv2d(1, 8, 3, padding=1)

        self.conv_e1 = nn.Conv2d(8, 16, 3, padding=1)
        self.maxpool_e1 = nn.MaxPool2d(2, stride=2)

        self.conv_e2 = nn.Conv2d(16, 32, 3, padding=1)

        self.conv_e3 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool_e3 = nn.MaxPool2d(2, stride=2)

        # context modules
        self.spatial_dropout = nn.Dropout2d(p=0.2)

        self.conv_c0 = nn.Conv2d(64,96,3, padding=1, dilation=1)
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
        self.conv_c12 = nn.Conv2d(96,ns,3, padding=1, dilation=1)

        # decoder convolutions # 150 x 150 x 16
        self.conv_d0 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d0 = nn.MaxPool2d(3, stride=3)

        self.conv_d1 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d1 = nn.MaxPool2d(3, stride=3)

        self.conv_d2 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d2 = nn.MaxPool2d(2, stride=2)

        self.conv_d3 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d3 = nn.MaxPool2d(2, stride=2)

        self.conv_d4 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d4 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv_d5 = nn.Conv2d(ns,ns,3, padding=1)
        self.maxpool_d5 = nn.MaxPool2d((1,2), stride=(1,2))

    # Input l: 600 x 600 x 1, v: ns x values_width
    def forward(self, l, v):
        # input
        v = v.transpose(0,1) # ensure v is ns x batch_size x values_width here         # [30, 2, 11]

        # (seq_len, batch, input_size)
        v, hn = self.lstm(v)                                    # [30, 2, 64]

        # Use only last output in the sequence produces
        v = v[-1]                                               # [2, 64]

        # encoder
        l = F.elu(self.conv_e0(l))                              # [2, 8, 600, 600]

        l = F.elu(self.conv_e1(l))                              # [2, 16, 600, 600]
        l = self.maxpool_e1(l)                                  # [2, 16, 300, 300]

        l = F.elu(self.conv_e2(l))                              # [2, 32, 300, 300]

        l = F.elu(self.conv_e3(l))                              # [2, 64, 300, 300]
        l = self.maxpool_e3(l)                                  # [2, 64, 150, 150]


        # v should have length=channels in l
        v = expand_biases(v, 150, 150)                          # [2, 64, 150, 150]
        l = torch.add(l, 1, v)                                  # [2, 64, 150, 150]
        l = self.spatial_dropout(F.elu(l))                      # [2, 64, 150, 150]


        #context module
        l = self.spatial_dropout(F.elu(self.conv_c0(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c1(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c2(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c3(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c4(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c5(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c6(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c7(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c8(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c9(l)))        # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c10(l)))       # [2, 96, 150, 150]
        l = self.spatial_dropout(F.elu(self.conv_c11(l)))       # [2, 96, 150, 150]
        l = F.elu(self.conv_c12(l))                             # [2, 30, 150, 150]

        # decoder convolutions
        l = F.elu(self.conv_d0(l))                              # [2, 30, 150, 150]
        l = self.maxpool_d0(l)                                  # [2, 30, 50, 50]

        l = F.elu(self.conv_d1(l))                              # [2, 30, 50, 50]
        l = self.maxpool_d1(l)                                  # [2, 30, 16, 16] TODO We are mising pixels here!

        l = F.elu(self.conv_d2(l))                              # [2, 30, 16, 16]
        l = self.maxpool_d2(l)                                  # [2, 30, 8, 8]

        l = F.elu(self.conv_d3(l))                              # [2, 30, 8, 8]
        l = self.maxpool_d3(l)                                  # [2, 30, 4, 4]

        l = F.elu(self.conv_d4(l))                              # [2, 30, 4, 4]
        l = self.maxpool_d4(l)                                  # [2, 30, 2, 2]

        l = F.elu(self.conv_d5(l))                              # [2, 30, 4, 2]
        l = self.maxpool_d5(l)                                  # [2, 30, 2, 1]

        l = l.squeeze()                                         # [2, 30, 2]
        l = l.view(-1, self.os)

        return l

def expand_biases(v, w, h):
    b = v.size(0) # batch size
    c = v.size(1) # number of channels
    return v.unsqueeze(2).expand(b,c,h).unsqueeze(3).expand(b,c,h,w)

def print_size(tensor, cond):
    if cond:
        print(tensor.size())
