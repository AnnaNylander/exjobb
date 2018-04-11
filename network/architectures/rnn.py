from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as mm
import numpy
import torch

''' Processes lidar and values for each time step to produce a path.
'''

class LSTMNet(nn.Module):
    ''' Using LSTM cells '''
    def __init__(self):
        super(LSTM, self).__init__()

        # LSTM architecture definitions
        input_size = 500 # The number of expected features in the input x
        hidden_size = 300 # The number of features in the hidden state h
        num_layers = 2 # Number of recurrent layers
        dropout = 0 # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        bidirectional = False # If True, becomes a bidirectional LSTM

        # encoder
        self.conv_e0 = nn.Conv2d(1,8,3, padding=1)
        self.bias_e0 = nn.Linear(330, 8, bias=false) # 330 x 8

        self.conv_e1 = nn.Conv2d(8,8,3, padding=1)
        self.maxpool_e1 = nn.MaxPool2d(2, stride=2)

        self.conv_e2 = nn.Conv2d(8,16,3, padding=1)

        self.conv_e3 = nn.Conv2d(16,16,3, padding=1)
        self.maxpool_e3 = nn.MaxPool2d(2, stride=2)

        # context modules
        self.spatial_dropout = nn.Dropout2d(p=0.2)

        self.conv_c0 = nn.Conv2d(16,96,3, padding=1, dilation=1)
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
        self.conv_c12 = nn.Conv2d(96,16,3, padding=1, dilation=1)

        # decoder convolutions # 150 x 150 x 16
        self.conv_d0 = nn.Conv2d(16,8,3, padding=1)
        self.maxpool_d0 = nn.Conv2d(3, stride=3)            # 50 x 50 x 16

        self.conv_d1 = nn.Conv2d(8,1,3, padding=1)
        self.maxpool_d1 = nn.Conv2d(2, stride=2)            # 25 x 25 x 16
        self.linear_d1 = nn.Linear(25*25 + 11,input_size)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # Initialize hidden state
        self.hidden = self.init_hidden()

    def forward(self, l, v):
        # input
        # TODO WHAT SHAPE TO USE HERE?
        v = v.view(-1, 30*11)                               # 330 x 1

        # encoder
        # TODO WHAT SHAPE TO USE HERE?
        l = F.elu(self.conv_e0(l))

        l = F.elu(self.conv_e1(l))           # 600 x 600 x 8
        l = self.maxpool_e1(l)                                # 300 x 300 x 8

        l = F.elu(self.conv_e2(l))           # 600 x 600 x 8

        l = F.elu(self.conv_e3(l))           # 600 x 600 x 8
        l = self.maxpool_e3(l)                                # 150 x 150 x 16

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
        l = self.maxpool_d0(l)                              # 50 x 50 x 8

        l = F.elu(self.conv_d1(l))
        l = self.maxpool_d1(l)                              # 25 x 25 x 1

        # Construct lstm input vector
        e = torch.cat(l, v)
        e = self.linear_d1(e)

        # Feed input into lstm to get output.
        # Output has shape (seq_len, batch, hidden_size * num_directions)
        output, self.hidden = self.lstm(e, self.hidden)

    def init_hidden():
        ''' Use this to reset the hidden state between inferences '''
        h_0 = Variable(torch.zeros(1, 1, self.hidden_size))
        c_0 = Variable(torch.zeros(1, 1, self.hidden_size))
        return (h_0, c_0)


class GRUNet(nn.Module):
    ''' Using GRU cells

        GET LSTM TO WORK FIRST, THEN IT SHOULD BE A MATTER OF COPY + PASTE HERE

    '''
    def __init__(self):
        super(CNNOnly, self).__init__()

    def forward(self, l, v):
        return l
