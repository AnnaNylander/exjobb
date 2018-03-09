# -*- coding: utf-8 -*-
import torch
import pandas
import numpy
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from MnistDataSet import MnistDataSet
from Network import Net

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, side, H, D_out = 20, 784, 28, 100, 10
kernel_size = 5
learning_rate = 1e-5

# Create Tensors to hold inputs and outputs, and wrap them in Variables
train_dataset = MnistDataSet(csv_file = 'data/mnist_train.csv')
test_dataset = MnistDataSet(csv_file = 'data/mnist_test.csv')
dataloader = DataLoader(train_dataset, batch_size=N, shuffle=True, num_workers=4)
dataloader_test = DataLoader(test_dataset)

# Construct our model by instantiating the class defined above
#model = torch.nn.Sequential(
    #torch.nn.Linear(D_in,H),
#    torch.nn.Conv2d(1,1,3), #as input it wants a 4d tensor with (batch size, channels in, img width, img height)
#    torch.nn.ReLU(),
#    torch.nn.Linear(26,H),
#    torch.nn.ReLU(),
#    torch.nn.Linear(H, D_out)
#)
#model = torch.load('saved/model.pt')
model = Net(side, H, D_out,kernel_size)
model.cuda()

#print(torch.cuda.is_available())

loss_fn = torch.nn.MultiLabelSoftMarginLoss()
loss_fn.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for batch_i, batch in enumerate(dataloader):


        images = (batch['image']).type(torch.cuda.FloatTensor)
        labels = (batch['label']).type(torch.cuda.FloatTensor)
        images = torch.unsqueeze(images, 1)
        x = Variable(images)
        y = Variable(labels)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        #if batch_i%N == 0:
            #print(batch_i, loss.data[0])

        # nollställ gradienterna
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    test_loss = 0
    for batch_i, batch in enumerate(dataloader_test):
        images = (batch['image']).type(torch.cuda.FloatTensor)
        labels = (batch['label']).type(torch.cuda.FloatTensor)
        images = torch.unsqueeze(images, 1)
        x = Variable(images)
        y = Variable(labels)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        test_loss = test_loss + loss.data[0]

    avg_loss = test_loss/len(test_dataset);
    print(avg_loss)

torch.save(model, 'saved/model3.pt')
