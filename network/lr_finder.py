# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math
from architectures import *
import argparse
from data_to_dict import getData
from dataset import OurDataset
import os
import shutil

parser = argparse.ArgumentParser(description='PyTorch Drive a car wohoo')
parser.add_argument('-a','--arch', default='', type=str, metavar='file.class',
                    help = 'Name of network to use. eg: LucaNetwork.LucaNet')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-e', '--epochs', default=10, type=int,
                    metavar='N', help='number of total epochs (default: 10)')

parser.add_argument('-d','--dataset', dest='dataset_path', default='',
                    type=str, metavar='PATH',
                    help = 'Name of folder in /media/annaochjacob/crucial/dataset/ ex \'Banana_split/\' (with trailing /)')
parser.add_argument('-s','--save-path', dest='save_path', default='',
                    type=str, metavar='PATH',
                    help = 'Name of folder in /media/annaochjacob/crucial/models/ ex \'SmallerNetwork1/\' (with trailing /)')

parser.add_argument('-o','--optim', default='SGD(model.parameters(), lr=1e-5, momentum=0.9, nesterov=True)', type=str,
                    metavar='name(model.parameters(), param**)',
                    help = 'optimizer and its param. Ex/default: \'SGD(model.parameters(), lr=1e-5, momentum=0.9, nesterov=True)\' )')
parser.add_argument('-pf', '--past-frames', default=0, type=int, dest='past_frames',
                    metavar='N', help='Number of past lidar frames provided to the network. (default: 0)')
parser.add_argument('-fs', '--frame-stride', default=1, type=int, dest='frame_stride',
                    metavar='N', help='Stride of past frames. Ex. past-frames=2 and frames-stride=2 where x is current frame'\
                     '\n gives x, x-2, x-4. (default: 1)')
parser.add_argument('-mpf','--manual_past_frames', default=None, type=str, metavar='\'1 2 3\'',
                    help = 'If not use past_frames and frames-stride, list which frames you want manually. Ex: \'1 3 5 7 10 13 16\'')

args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'
PATH_SAVE = PATH_BASE + 'models/' + args.save_path
PATH_DATA = PATH_BASE + 'dataset/' + args.dataset_path
NUM_WORKERS = 3
PIN_MEM = False

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
    print(len(trn_loader))

    for batch in trn_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))
        values = Variable((batch['value']).type(torch.cuda.FloatTensor))
        targets = Variable((batch['output']).type(torch.cuda.FloatTensor))

        lidars = lidars.view(-1, args.past_frames+1, 600, 600)
        values = values.view(-1, 30, 11)
        targets = targets.view(-1, 60)

        optimizer.zero_grad()
        outputs = net(lidars,values)
        loss = criterion(outputs, targets)
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
        if batch_num % 10 == 0:
            print('Batch: %i \tLoss: %.3f \tlr: %.3e' %(batch_num,smoothed_loss,lr))
    return log_lrs, losses


def main():

    for i in range(1):
        model = eval(args.arch + "()")
        model.cuda()

        super_train = {}

        if args.manual_past_frames is None:
            args.manual_past_frames = list(range(args.frame_stride,args.frame_stride*args.past_frames+1, args.frame_stride))


        # Create a dictionary containing paths to data in all smaller data sets
        for subdir in os.listdir(PATH_DATA):
            subpath = PATH_DATA + subdir + '/'
            train_data = getData(subpath + 'train/', args.manual_past_frames)
            for key in list(train_data.keys()):
                if key in super_train:
                    super_train[key] = np.concatenate((super_train[key], train_data[key]), axis=0)
                else:
                    super_train[key] = train_data[key]

        train_dataset = OurDataset(super_train) #14000
        trn_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEM)



        optimizer = eval('torch.optim.' + args.optim)
        criterion = torch.nn.MSELoss().cuda()

        log_lrs, losses = find_lr(model, trn_loader, optimizer, criterion)
        plt.plot(log_lrs,losses)

    plt.show()

if __name__ == '__main__':
    main()
