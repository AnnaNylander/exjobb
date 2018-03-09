# -*- coding: utf-8 -*-
import torch
import pandas
import numpy
import argparse
import time
import os
import shutil
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from data_to_dict import getTestSet,getTrainingSet,getTrainingBatch
from dataset import CifarDataSet
from network import First_Network, Network
from result_meter import ResultMeter

# NOTE! Validate during traning. Test is last when model finished traning.

parser = argparse.ArgumentParser(description='PyTorch Cifar-10 classification')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-e', '--epochs', default=10, type=int,
                    metavar='N', help='number of total epochs (default: 10)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')


# TODO variate learning_rate depending on epoch.
# TODO save checkpoints
# TODO continue to train on checkpoint.
# TODO more things to measure??

def main():
    # variables
    global args
    args = parser.parse_args()
    best_res = 10000 #big number
    D_in, D_out = 32*32, 10
    learning_rate = 1e-4
    epoch_start = 0

    # load all time best
    print("-----Load all time best loss (for comparision)-----")
    all_time_best_res = 10000 # big number
    all_time_best = load_checkpoint('saved/all_time_best.pt')
    if all_time_best is not None:
        all_time_best_res = all_time_best['best_res']
        print("avg loss @:",all_time_best_res)

    # create model
    print("-----Creating network-----")
    model = Network()
    model.cuda()

    # define loss function and optimizer
    print("-----Creating lossfunction and optimizer-----")
    loss_fn = torch.nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #resume from checkpoint
    if args.resume:
        print("Resume from checkpoint:")
        checkpoint = load_checkpoint(args.resume)
        if checkpoint is None:
            print("No file found. Exiting...")
            return
        epoch_start = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_res = checkpoint['best_res']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint sucessfully")

    # Load datasets
    print("-----Loading datasets-----")
    train_dataset = CifarDataSet(getTrainingSet())
    test_dataset = CifarDataSet(getTestSet())

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=4)
    dataloader_val = DataLoader(test_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=4)

    if args.evaluate:
        print("_____EVALUATE MODEL______")
        validate(model, dataloader_val, loss_fn)
        return

    #train network
    print("______TRAIN MODEL_______")
    print("train network for a total of {diff} [{epochs}/{total_epochs}]"
            " epochs.".format(diff = max(args.epochs-epoch_start,0),
            epochs = epoch_start, total_epochs = args.epochs))
    for epoch in range(epoch_start,args.epochs):

        # train for one epoch
        train(model,dataloader,loss_fn, optimizer, epoch)

        # evaluate on validation set
        res = validate(model, dataloader_val, loss_fn)

        #remember best and save checkpoint
        is_best = res < best_res
        best_res = min(res, best_res)
        is_all_time_best = res < all_time_best_res
        all_time_best_res = min(res, all_time_best_res)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_res': best_res,
            'optimizer' : optimizer.state_dict(),
        }, is_best, is_all_time_best)


def train(model, dataloader, loss_fn, optimizer, epoch):
    model.train() # switch to train mode

    losses = ResultMeter()
    batch_time = ResultMeter()

    start = time.time()
    for i, batch in enumerate(dataloader):

        inputs = Variable((batch['image']).type(torch.cuda.FloatTensor))
        targets = Variable((batch['label']).type(torch.cuda.FloatTensor))

        output = model(inputs)
        loss = loss_fn(output, targets)

        optimizer.zero_grad() # reset gradients
        loss.backward()
        optimizer.step() # update weights

        # document result
        losses.update(loss.data[0])
        batch_time.update(time.time() - start)
        start = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format( epoch+1,
                   i, len(dataloader), batch_time=batch_time, losses=losses))

def validate(model, dataloader, loss_fn):
    model.eval() # switch to eval mode

    losses = ResultMeter()
    batch_time = ResultMeter()

    start = time.time()
    for i, batch in enumerate(dataloader):
        inputs = Variable((batch['image']).type(torch.cuda.FloatTensor))
        targets = Variable((batch['label']).type(torch.cuda.FloatTensor))

        output = model(inputs)
        loss = loss_fn(output, targets)

        # document result
        losses.update(loss.data[0])
        batch_time.update(time.time() - start)
        start = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}] \t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
                   i, len(dataloader), batch_time=batch_time, losses=losses))

    print('Validation complete. Final results: \n'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
            batch_time = batch_time, losses=losses ))

    return losses.avg


def save_checkpoint(state, is_best, is_all_time_best,
        filename = 'saved/checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        print("Best thus far!")
        shutil.copyfile(filename, 'saved/best.pt')
    if is_all_time_best:
        print("ALL TIME BEST! GOOD JOB!")
        shutil.copyfile(filename, 'saved/all_time_best.pt')
    print("\n")

def load_checkpoint(filename):
    if os.path.isfile(filename):
        print("Loading model at '{}'".format(filename))
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print("No file found at '{}'".format(filename))
        return None

def accuracy(output, target):
    """ Not implemented yet. Unsure exactly what we want."""

if __name__ == '__main__':
    main()
