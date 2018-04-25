from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as mm
import numpy
import torch

PRINT = False

''' Processes lidar and values for each time step to produce a path.
'''

class LSTMNet(nn.Module):
    ''' Using LSTM cells '''
    def __init__(self):
        super(LSTMNet, self).__init__()

        # LSTM architecture definitions
        self.bptt = 3 #Including the current time step
        self.input_size = 636 # The number of expected features in the input x
        self.hidden_size = 300 # The number of features in the hidden state h
        self.num_layers = 2 # Number of recurrent layers
        self.dropout = 0 # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        self.bidirectional = False # If True, becomes a bidirectional LSTM
        self.h_n = None
        self.c_n = None

        # lidar encoder
        self.conv_e0 = nn.Conv2d(self.bptt, 32, 3, padding=1)

        self.conv_e1 = nn.Conv2d(32, 32, 3, padding=1)
        self.maxpool_e1 = nn.MaxPool2d(2, stride=2)

        self.conv_e2 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv_e3 = nn.Conv2d(64, 64, 3, padding=1)
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
        self.conv_c12 = nn.Conv2d(96,64,3, padding=1, dilation=1)

        # decoder convolutions # 150 x 150 x 32
        self.conv_d0 = nn.Conv2d(64,32,3, padding=1)
        self.maxpool_d0 = nn.MaxPool2d(3, stride=3)

        self.conv_d1 = nn.Conv2d(32,16,3, padding=1)
        self.maxpool_d1 = nn.MaxPool2d(2, stride=2)

        self.conv_d2 = nn.Conv2d(16, self.bptt, 3, padding=1)

        self.lstm_d3 = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.linear_d3 = nn.Linear(300,60)

    def forward(self, l, v):

        # l is expected to be of shape [2, self.bptt, 600, 600]
        # v is expected to be of shape [2, self.bptt, 11]
        # h_n is expected to be of shape [2*1, 2, 300]
        # c_n is expected to be of shape [2*1, 2, 300]

        # encoder
        l = F.elu(self.conv_e0(l))                              # [2, 32, 600, 600]

        l = F.elu(self.conv_e1(l))                              # [2, 32, 600, 600]
        l = self.maxpool_e1(l)                                  # [2, 32, 300, 300]

        l = F.elu(self.conv_e2(l))                              # [2, 64, 300, 300]

        l = F.elu(self.conv_e3(l))                              # [2, 64, 300, 300]
        l = self.maxpool_e3(l)                                  # [2, 64, 150, 150]

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
        l = self.spatial_dropout(F.elu(self.conv_c11(l)))       # [2, 96, 150, 150]
        l = F.elu(self.conv_c12(l))                             # [2, 64, 150, 150]

        # decoder convolutions
        l = F.elu(self.conv_d0(l))                              # [2, 64, 150, 150]
        l = self.maxpool_d0(l)                                  # [2, 64, 50, 50]

        l = F.elu(self.conv_d1(l))                              # [2, 32, 50, 50]
        l = self.maxpool_d1(l)                                  # [2, 32, 25, 25]

        l = F.elu(self.conv_d2(l))                              # [2, self.bptt, 25, 25]

        # Make a 1D vector of the 2D image
        l = l.view(l.size(0), self.bptt, -1)

        # Pick out values for the most recent time steps
        # NOTE stride is always 1 here!
        v = v[:,0:self.bptt,:]

        # Construct lstm input vectors by concatenating the values to the lidar
        e = torch.cat((l, v),2)

        # change order to (seq, batch, input)
        e = e.transpose(0,1)

        # Feed input into lstm to get output.
        # Output y has shape (seq_len, batch, hidden_size * num_directions)
        if self.h_n is None:
            self.h_n = Variable(torch.zeros(self.num_layers, l.size(0), self.hidden_size).cuda())
            self.c_n = Variable(torch.zeros(self.num_layers, l.size(0), self.hidden_size).cuda())

        y, (h_n, c_n) = self.lstm_d3(e, (self.h_n, self.c_n))                  # [30, 2, 300]'

        # Detach to forget gradients of past hidden and cell states
        self.h_n = h_n.detach()
        self.c_n = c_n.detach()

        y = self.linear_d3(y[-1])                               # [2, 60]

        return y

class LSTMNetBi(nn.Module):
    ''' Using bidirectional LSTM cells '''
    def __init__(self):
        super(LSTMNetBi, self).__init__()

        # LSTM architecture definitions
        self.bptt = 3
        self.input_size = 636 # The number of expected features in the input x
        self.hidden_size = 300 # The number of features in the hidden state h
        self.num_layers = 2 # Number of recurrent layers
        self.dropout = 0 # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        self.bidirectional = True # If True, becomes a bidirectional LSTM
        self.h_n = None
        self.c_n = None

        # lidar encoder
        self.conv_e0 = nn.Conv2d(self.bptt, 32, 3, padding=1)

        self.conv_e1 = nn.Conv2d(32, 32, 3, padding=1)
        self.maxpool_e1 = nn.MaxPool2d(2, stride=2)

        self.conv_e2 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv_e3 = nn.Conv2d(64, 64, 3, padding=1)
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
        self.conv_c12 = nn.Conv2d(96,64,3, padding=1, dilation=1)

        # decoder convolutions # 150 x 150 x 32
        self.conv_d0 = nn.Conv2d(64,32,3, padding=1)
        self.maxpool_d0 = nn.MaxPool2d(3, stride=3)

        self.conv_d1 = nn.Conv2d(32,16,3, padding=1)
        self.maxpool_d1 = nn.MaxPool2d(2, stride=2)

        self.conv_d2 = nn.Conv2d(16, self.bptt, 3, padding=1)

        self.lstm_d3 = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.linear_d3 = nn.Linear(600,60)

    def forward(self, l, v):

        # l is expected to be of shape [2, self.bptt, 600, 600]
        # v is expected to be of shape [2, self.bptt, 11]
        # h_n is expected to be of shape [2*1, 2, 300]
        # c_n is expected to be of shape [2*1, 2, 300]

        # encoder
        l = F.elu(self.conv_e0(l))                              # [2, 32, 600, 600]

        l = F.elu(self.conv_e1(l))                              # [2, 32, 600, 600]
        l = self.maxpool_e1(l)                                  # [2, 32, 300, 300]

        l = F.elu(self.conv_e2(l))                              # [2, 64, 300, 300]

        l = F.elu(self.conv_e3(l))                              # [2, 64, 300, 300]
        l = self.maxpool_e3(l)                                  # [2, 64, 150, 150]

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
        l = self.spatial_dropout(F.elu(self.conv_c11(l)))       # [2, 96, 150, 150]
        l = F.elu(self.conv_c12(l))                             # [2, 64, 150, 150]

        # decoder convolutions
        l = F.elu(self.conv_d0(l))                              # [2, 64, 150, 150]
        l = self.maxpool_d0(l)                                  # [2, 64, 50, 50]

        l = F.elu(self.conv_d1(l))                              # [2, 32, 50, 50]
        l = self.maxpool_d1(l)                                  # [2, 32, 25, 25]

        l = F.elu(self.conv_d2(l))                              # [2, 30, 25, 25]

        l = l.view(l.size(0), self.bptt ,-1)

        # Make a 1D vector of the 2D image
        l = l.view(l.size(0), self.bptt, -1)

        # Pick out values for the most recent time steps
        # NOTE stride is always 1 here!
        v = v[:,0:self.bptt,:]

        # Construct lstm input vectors by concatenating the values to the lidar
        e = torch.cat((l, v),2)

        # change order to (seq, batch, input)
        e = e.transpose(0,1)

        # Feed input into lstm to get output.
        # Output y has shape (seq_len, batch, hidden_size * num_directions)
        if self.h_n is None:
            self.h_n = Variable(torch.zeros(self.num_layers*2, l.size(0), self.hidden_size).cuda())
            self.c_n = Variable(torch.zeros(self.num_layers*2, l.size(0), self.hidden_size).cuda())

        y, (h_n, c_n) = self.lstm_d3(e, (self.h_n, self.c_n))                  # [30, 2, 300]'

        # Detach to forget gradients of past hidden and cell states
        self.h_n = h_n.detach()
        self.c_n = c_n.detach()

        y = self.linear_d3(y[-1])                               # [2, 60]

        return y

class GRUNet(nn.Module):
    ''' Using bidirectional LSTM cells '''
    def __init__(self):
        super(GRUNet, self).__init__()

        # LSTM architecture definitions
        self.bptt = 3
        self.input_size = 636 # The number of expected features in the input x
        self.hidden_size = 300 # The number of features in the hidden state h
        self.num_layers = 2 # Number of recurrent layers
        self.dropout = 0 # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        self.bidirectional = False # If True, becomes a bidirectional LSTM
        self.h_n = None

        # lidar encoder
        self.conv_e0 = nn.Conv2d(self.bptt, 32, 3, padding=1)

        self.conv_e1 = nn.Conv2d(32, 32, 3, padding=1)
        self.maxpool_e1 = nn.MaxPool2d(2, stride=2)

        self.conv_e2 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv_e3 = nn.Conv2d(64, 64, 3, padding=1)
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
        self.conv_c12 = nn.Conv2d(96,64,3, padding=1, dilation=1)

        # decoder convolutions # 150 x 150 x 32
        self.conv_d0 = nn.Conv2d(64,32,3, padding=1)
        self.maxpool_d0 = nn.MaxPool2d(3, stride=3)

        self.conv_d1 = nn.Conv2d(32,16,3, padding=1)
        self.maxpool_d1 = nn.MaxPool2d(2, stride=2)

        self.conv_d2 = nn.Conv2d(16, self.bptt, 3, padding=1)

        self.gru_d3 = nn.GRU(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.linear_d3 = nn.Linear(300,60)

    def forward(self, l, v):

        # l is expected to be of shape [2, self.bptt, 600, 600]
        # v is expected to be of shape [2, self.bptt, 11]
        # h_n is expected to be of shape [2*1, 2, 300]
        # c_n is expected to be of shape [2*1, 2, 300]

        # encoder
        l = F.elu(self.conv_e0(l))                              # [2, 32, 600, 600]

        l = F.elu(self.conv_e1(l))                              # [2, 32, 600, 600]
        l = self.maxpool_e1(l)                                  # [2, 32, 300, 300]

        l = F.elu(self.conv_e2(l))                              # [2, 64, 300, 300]

        l = F.elu(self.conv_e3(l))                              # [2, 64, 300, 300]
        l = self.maxpool_e3(l)                                  # [2, 64, 150, 150]

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
        l = self.spatial_dropout(F.elu(self.conv_c11(l)))       # [2, 96, 150, 150]
        l = F.elu(self.conv_c12(l))                             # [2, 64, 150, 150]

        # decoder convolutions
        l = F.elu(self.conv_d0(l))                              # [2, 64, 150, 150]
        l = self.maxpool_d0(l)                                  # [2, 64, 50, 50]

        l = F.elu(self.conv_d1(l))                              # [2, 32, 50, 50]
        l = self.maxpool_d1(l)                                  # [2, 32, 25, 25]

        l = F.elu(self.conv_d2(l))                              # [2, 30, 25, 25]

        # Make a 1D vector of the 2D image
        l = l.view(l.size(0), self.bptt, -1)

        # Pick out values for the most recent time steps
        # NOTE stride is always 1 here!
        v = v[:,0:self.bptt,:]

        # Construct lstm input vectors by concatenating the values to the lidar
        e = torch.cat((l, v),2)

        # change order to (seq, batch, input)
        e = e.transpose(0,1)

        # Feed input into lstm to get output.
        # Output y has shape (seq_len, batch, hidden_size * num_directions)
        if self.h_n is None:
            self.h_n = Variable(torch.zeros(self.num_layers, l.size(0), self.hidden_size).cuda())

        y, h_n = self.gru_d3(e, self.h_n)                  # [30, 2, 300]'

        # Detach to forget gradients of past hidden states
        self.h_n = h_n.detach()

        y = self.linear_d3(y[-1])                               # [2, 60]

        return y

def expand_biases(v, w, h):
    b = v.size(0) # batch size
    c = v.size(1) # number of channels
    return v.unsqueeze(2).expand(b,c,h).unsqueeze(3).expand(b,c,h,w)

def print_size(tensor, cond):
    if cond:
        print(tensor.size())
