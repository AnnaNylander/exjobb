# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math

BEST_LR = 3*1e-6

def main():

    net1 = nn.Sequential(
          nn.Linear(100,50),
          nn.ReLU(),
          nn.Linear(50,25),
          nn.ReLU(),
          nn.Linear(25,1),
          nn.ReLU()
        )

    dataset = FakeDataset(100000)
    trn_loader = DataLoader(dataset,batch_size=10)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr = 0.01, momentum=0.9)
    criterion = torch.nn.MSELoss()

    lrs, losses = train(net1, trn_loader, optimizer1, criterion, min_value=BEST_LR/10, max_value=BEST_LR)
    #plt.plot(lrs)
    #plt.figure()
    plot_1, = plt.plot(losses)
    plot_1.set_label('Scheduled')
    #plt.show()

    net2 = nn.Sequential(
          nn.Linear(100,50),
          nn.ReLU(),
          nn.Linear(50,1),
          nn.ReLU()
        )
    optimizer2 = torch.optim.SGD(net2.parameters(), lr = 0.01, momentum=0.9)
    lrs, losses = train(net2, trn_loader, optimizer2, criterion, min_value=BEST_LR, max_value=BEST_LR)
    #plt.plot(lrs)
    #plt.figure()
    plot_2, = plt.plot(losses)
    plot_2.set_label('Fixed')
    plt.legend((plot_1,plot_2), ('Scheduled','Fixed'))
    plt.show()

class FakeDataset(Dataset):

    def __init__(self, length, transform=None):
        Dataset.__init__(self)
        self.inputs = []
        self.targets = []

        for i in range(0,length):
            mu = np.random.randint(10)
            self.inputs.append(np.random.normal(mu, 1, 100))
            self.targets.append([mu])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input = torch.FloatTensor(self.inputs[idx])
        target = torch.FloatTensor(self.targets[idx])
        return input, target

def train(net, trn_loader, optimizer, criterion, min_value=1e-8, max_value=1e-5, beta=0.98, n_batches=5000):
    num = len(trn_loader)-1
    #mult = (final_value / init_value) ** (1/num)
    lr_diff = 2*(max_value - min_value)/n_batches
    lr = min_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    lrs = []
    for data in trn_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > n_batches:
            return lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        lrs.append(lr)
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        if batch_num <= n_batches/2:
            lr += lr_diff
        else:
            lr -= lr_diff

        optimizer.param_groups[0]['lr'] = lr
        if batch_num % 100 == 0:
            print('Batch', batch_num, 'done')

    return lrs, losses

if __name__ == '__main__':
    main()
