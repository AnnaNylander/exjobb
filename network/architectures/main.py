from cnn_bias import CNNBiasFirst
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def main():


    # Create fake data
    batch_size = 2
    lidar_data = torch.ones(batch_size, 1, 600, 600)
    value_data = torch.ones(batch_size, 30, 11)
    lidars = Variable(lidar_data.type(torch.FloatTensor))
    values = Variable(value_data.type(torch.FloatTensor))
    #print(lidars.size())
    #print(values.size())

    # Instantiate model
    model = CNNBiasFirst()

    model(lidars, values)


    #print('DONE')

if __name__ == '__main__':
    main()
