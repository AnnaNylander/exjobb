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
from data_to_dict import get_data
from dataset import OurDataset
import os
import re
import shutil
from main import get_data_loader

parser = argparse.ArgumentParser(description='PyTorch Drive a car wohoo')
parser.add_argument('-a','--arch', default='', type=str, metavar='file.class',
                    help = 'Name of network to use. eg: LucaNetwork.LucaNet')
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                    help='Whether to shuffle training data or not. (default: False)')
parser.add_argument('--no-intention', dest='no_intention', action='store_true',
                    help='Set all intentions to 0. (default: False (aka keep intentions))')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-e', '--epochs', default=10, type=int,
                    metavar='N', help='number of total epochs (default: 10)')
parser.add_argument('-p','--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('-pl', '--plot-freq', default=100, type=int,
                    metavar='N', dest='plot_freq', help='plot frequency (default: 100 batch)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Name of folder in /media/annaochjacob/crucial/models/ ex \'SmallerNetwork1/checkpoint.pt\' ')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--scheduler', dest='scheduler', action='store_true',
                    help='Whether to manually adjust learning rate as we train. (https://sgugger.github.io/the-1cycle-policy.html)')
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
                    metavar='N', help='Number of past lidar frames provided to the network (For RNN it is bptt) (default: 0)')
parser.add_argument('-fs', '--frame-stride', default=1, type=int, dest='frame_stride',
                    metavar='N', help='Stride of past frames. Ex. past-frames=2 and frames-stride=2 where x is current frame'\
                     '\n gives x, x-2, x-4. (default: 1)')
parser.add_argument('-mpf','--manual_past_frames', default=None, type=str, metavar='\'1 2 3\'',
                    help = 'If not use past_frames and frames-stride, list which frames you want manually. Ex: \'1 3 5 7 10 13 16\''\
                    'NOTE: Not applicable for RNNs!! Use -pf and -fs flags instead.')
parser.add_argument('-bptt', '--bptt', default=1, type=int, dest='bptt',
                    metavar='N', help='Back propagation through time. Option only available for RNNs. (default = 1)')
# NOTE: Currently we find all rnns by doing regex. If this changes to be true, add this argument.
parser.add_argument('-rnn', '--rnn', dest='rnn', action='store_true',
                    help='Wheter we have an rnn or not. (not needed if arch str contains \'rnn\')')
parser.add_argument('-bl', '--balance', dest='balance', action='store_true',
                    help='Balance dataset by sampling with replacement. Not applicable for RNNs. Forces shuffle to True in training set.')

#TODO: data_to_dict, dataset, main.
# save, load,
args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'
PATH_RESUME = PATH_BASE + 'models/' + args.resume
PATH_SAVE = PATH_BASE + 'models/' + args.save_path
if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)
PATH_DATA = PATH_BASE + 'dataset/' + args.dataset_path
NUM_WORKERS = 3
PIN_MEM = False

if args.manual_past_frames:
    args.manual_past_frames = [int(i) for i in args.manual_past_frames.split(' ')]

rnn_arch_match = re.search('RNN', args.arch, flags=re.IGNORECASE)
if rnn_arch_match is not None:
    args.rnn = True

def find_lr(net, trn_loader, optimizer, criterion, init_value = 1e-8, final_value=10., beta = 0.98, sampler_max = None):
    if sampler_max is not None:
        num = int(sampler_max/args.batch_size) + 1
    else:
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

        optimizer.zero_grad()
        outputs = net(lidars,values)
        loss = criterion(outputs, targets)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 1000 * best_loss:
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
    #write info file
    if not os.path.exists(PATH_SAVE):
            os.makedirs(PATH_SAVE)
    write_info_file()

    for i in range(1):
        model = eval(args.arch + "()")
        model.cuda()
        if not args.rnn:
            sampler_max = 100000
        else:
            sampler_max = None
        trn_loader = get_data_loader(PATH_DATA + 'train/', shuffle=args.shuffle, balance=args.balance, sampler_max = sampler_max)

        optimizer = eval('torch.optim.' + args.optim)
        criterion = torch.nn.MSELoss().cuda()

        log_lrs, losses = find_lr(model, trn_loader, optimizer, criterion, sampler_max = sampler_max)
        plt.plot(log_lrs,losses)

    #plt.show()
    plt.savefig(PATH_SAVE + 'lr_finder.png')
    write_loss_file(log_lrs, losses)

def write_loss_file(log_lrs, losses):
    np.savetxt(PATH_SAVE + "log_lrs.txt", log_lrs, comments='', delimiter=',',fmt='%.8f')
    np.savetxt(PATH_SAVE + "losses.txt", losses, comments='', delimiter=',',fmt='%.8f')

def write_info_file():
    info = ""
    for key in args.__dict__:
        info += str(key) + " : " + str(args.__dict__[key]) + "\n"

    file = open(PATH_SAVE + "info.txt", "w")
    file.write(info)
    file.close()

if __name__ == '__main__':
    main()
