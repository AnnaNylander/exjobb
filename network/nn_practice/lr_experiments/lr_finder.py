# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math

def find_lr(net, trn_loader, optimizer, criterion, init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
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
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        if batch_num % 100 == 0:
            print('Batch', batch_num, 'done')
    return log_lrs, losses


def main():

    for i in range(5):
        net = nn.Sequential(
              nn.Linear(100,50),
              nn.ReLU(),
              nn.Linear(50,1),
              nn.ReLU()
            )

        dataset = FakeDataset(100000)
        trn_loader = DataLoader(dataset,batch_size=100)
        optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
        criterion = torch.nn.MSELoss()

        log_lrs, losses = find_lr(net, trn_loader, optimizer, criterion)
        plt.plot(log_lrs,losses)

    plt.show()

class FakeDataset(Dataset):

    def __init__(self, length, transform=None):
        Dataset.__init__(self)
        self.inputs = []
        self.targets = []

        for i in range(0,length):
            mu = np.random.randint(10)
            self.inputs.append(np.random.normal(mu, 0.5, 100))
            self.targets.append([mu])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input = torch.FloatTensor(self.inputs[idx])
        target = torch.FloatTensor(self.targets[idx])
        return input, target

if __name__ == '__main__':
    main()
