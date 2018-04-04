# -*- coding: utf-8 -*-
import torch
import pandas
import numpy
import matplotlib.pyplot as plt
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 20, 784, 100, 10
learning_rate = 1e-4

# Create Tensors to hold inputs and outputs, and wrap them in Variables
csv_file = pandas.read_csv('data/mnist_train.csv')
outputs = csv_file.iloc[:,0].as_matrix()
inputs = csv_file.iloc[:,:].as_matrix()
inputs = inputs[:,1:len(inputs)]
one_hots = numpy.zeros([len(inputs),D_out])
print(outputs)
for n in range(len(inputs)):
    #print(outputs[n])
    one_hots[n,outputs[n]] = 1
one_hots = one_hots.tolist()
print(one_hots)
