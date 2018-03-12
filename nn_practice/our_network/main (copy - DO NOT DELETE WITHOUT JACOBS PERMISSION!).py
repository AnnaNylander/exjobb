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
from data_to_dict import getData
from dataset import OurDataset
from network import Network
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
parser.add_argument('--plot-freq', '-pl', default=100, type=int,
                    metavar='N', dest='plot_freq', help='plot frequency (default: 10)')
parser.add_argument('--dataset', dest='dataset_path', default='./dataset/',
                    type=str, metavar='PATH', help = 'path to dataset folder.')
parser.add_argument('--save-path', dest='save_path', default='./saved/',
                    type=str, metavar='PATH', help = 'path to where to save models.')

args = parser.parse_args()
# TODO variate learning_rate depending on epoch.
# TODO save checkpoints
# TODO continue to train on checkpoint.
# TODO more things to measure??

def main():
    # variables
    best_res = 10000 #big number
    learning_rate = 1e-3 #1e-4
    epoch_start = 0

    # load all time best
    print("-----Load all time best loss (for comparision)-----")
    all_time_best_res = 10000 # big number
    all_time_best = load_checkpoint( args.save_path + 'all_time_best.pt')
    if all_time_best is not None:
        all_time_best_res = all_time_best['best_res']
        print("avg loss @:",all_time_best_res)
        del all_time_best # Remove from GPU memory

    # create model
    print("-----Creating network-----")
    model = Network()
    model.cuda()
    print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))

    # define loss function and optimizer
    print("-----Creating lossfunction and optimizer-----")
    loss_fn = torch.nn.MSELoss().cuda()
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
    train_dataset = OurDataset(getData(args.dataset_path + 'train/',100)) # 2100
    validate_dataset = OurDataset(getData(args.dataset_path + 'validate/',50)) # 600
    test_dataset = OurDataset(getData(args.dataset_path + 'test/',50)) # 300

    dataloader_train= DataLoader(train_dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(validate_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)

    if args.evaluate:
        print("_____EVALUATE MODEL______")
        validate(model, dataloader_test, loss_fn)
        return

    #train network
    print("______TRAIN MODEL_______")
    print("train network for a total of {diff} [{epochs}/{total_epochs}]"
            " epochs.".format(diff = max(args.epochs-epoch_start,0),
            epochs = epoch_start, total_epochs = args.epochs))

    for epoch in range(epoch_start,args.epochs):

        # train for one epoch
        avg_loss = train(model,dataloader_train,loss_fn, optimizer, epoch)
        save_statistic(avg_loss, epoch, args.save_path + 'training.csv')

        # evaluate on validation set
        res = validate(model, dataloader_val, loss_fn)
        save_statistic(res, epoch, args.save_path + 'validation.csv')

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

    #final evaluation on test dataset
    print("_____EVALUATE MODEL______")
    validate(model, dataloader_test, loss_fn)


def train(model, dataloader, loss_fn, optimizer, epoch):
    model.train() # switch to train mode

    losses = ResultMeter()
    batch_time = ResultMeter()

    start = time.time()
    for i, batch in enumerate(dataloader):
#        print(i)
#        print(batch)
#        print('helo')
        lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))
        values = Variable((batch['value']).type(torch.cuda.FloatTensor))
        targets = Variable((batch['output']).type(torch.cuda.FloatTensor))
#        print(numpy.shape(targets))
        lidars = lidars.view(-1, 1, 600, 600)
        values = values.view(-1, 1, 30, 11)
        targets = targets.view(-1, 60)
#        print(numpy.shape(lidars))
#        print(numpy.shape(values))
#        print(numpy.shape(targets))

        output = model(lidars, values)
        loss = loss_fn(output, targets)
#        print(output)
        optimizer.zero_grad() # reset gradients
        loss.backward()
        optimizer.step() # update weights
        # document result
        losses.update(loss.data[0])
        batch_time.update(time.time() - start)
        start = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format( epoch+1,
                   i, len(dataloader), batch_time=batch_time, losses=losses))

    return losses.avg

def validate(model, dataloader, loss_fn):
    model.eval() # switch to eval mode

    losses = ResultMeter()
    batch_time = ResultMeter()

    start = time.time()
    for i, batch in enumerate(dataloader):
        lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor),volatile=True)
        values = Variable((batch['value']).type(torch.cuda.FloatTensor),volatile=True)
        targets = Variable((batch['output']).type(torch.cuda.FloatTensor),volatile=True)
        #targets = targets.view(-1, 60)
        lidars = lidars.view(-1, 1, 600, 600)
        values = values.view(-1, 1, 30, 11)
        targets = targets.view(-1, 60)

        output = model(lidars, values)
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
        filename = args.save_path + 'checkpoint.pt'):
    torch.save(state, filename)
    #if is_best:
    #    print("Best so far!")
    #    shutil.copyfile(filename, args.save_path + 'best.pt')
    if is_all_time_best:
        print("ALL TIME BEST! GOOD JOB!")
        shutil.copyfile(filename, args.save_path + 'all_time_best.pt')
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

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        #print(p.size())
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def save_statistic(statistic, epoch, path):
    fd = open(path,'a')
    row = [str(epoch),str(statistic)]
    fd.write(','.join(row) + '\n')
    fd.close()

if __name__ == '__main__':
    main()
