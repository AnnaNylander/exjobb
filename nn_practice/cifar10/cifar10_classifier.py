# -*- coding: utf-8 -*-
import torch
import pandas
import numpy
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from data_to_dict import getTestSet,getTrainingSet,getTrainingBatch
from dataset import CifarDataSet
from network import Network

# NOTE! Validate during traning. Test is last when model finished traning.

# TODO variate learning_rate depending on epoch.
# TODO save checkpoints
# TODO continue to train on checkpoint.
# TODO more things to measure??

def main():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, side, H, D_out = 20, 32*32, 32, 100, 10
    kernel_size = 5
    epochs = 10
    learning_rate = 1e-5

    # Load datasets
    train_dataset = CifarDataSet(getTrainingSet())
    test_dataset = CifarDataSet(getTestSet())
    dataloader = DataLoader(train_dataset, batch_size=N, shuffle=True, num_workers=4)
    dataloader_test = DataLoader(test_dataset)

    # create model
    model = Network(side, H, D_out,kernel_size)
    model.cuda()

    # loss function
    loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    loss_fn.cuda()

    # how we update weights
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        # train for one epoch
        train(model,dataloader,loss_fn, optimizer)

        # evaluate on validation set
        val_prediction = validate(model, dataloader_test, loss_fn)



def train(model, dataloader, loss_fn, optimizer):
    model.train() # switch to train mode
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad() # reset gradients

        inputs = Variable((batch['image']).type(torch.cuda.FloatTensor))
        targets = Variable((batch['label']).type(torch.cuda.FloatTensor))
        prediction = model(inputs)

        loss = loss_fn(prediction, targets)
        loss.backward()

        optimizer.step() # update weights

def validate(model, dataloader, loss_fn):
    model.eval() # switch to eval mode
    test_loss = 0
    for i, batch in enumerate(dataloader):
        inputs = Variable((batch['image']).type(torch.cuda.FloatTensor))
        targets = Variable((batch['label']).type(torch.cuda.FloatTensor))

        prediction = model(inputs)
        loss = loss_fn(prediction, targets)

        test_loss = test_loss + loss.data[0]

    avg_loss = test_loss/10000;
    print(avg_loss)
    return avg_loss


def save_checkpoint():
    torch.save(model, 'saved/cifar_model_1.pt')

if __name__ == '__main__':
    main()
