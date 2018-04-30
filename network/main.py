# -*- coding: utf-8 -*-
import torch
import pandas
import numpy
import argparse
import time
import os
import re
import shutil
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, WeightedRandomSampler

from architectures import *
from data_to_dict import get_data
from dataset import OurDataset, RNNDataset
from scheduler import Scheduler
#from architectures.network import LucaNetwork, SmallerNetwork1, SmallerNetwork2
from result_meter import ResultMeter

# NOTE! Validate during traning. Test is last when model finished traning.

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
# NOTE not used at the moment
#parser.add_argument('-bptt', '--bptt', default=1, type=int, dest='bptt',
#                    metavar='N', help='Back propagation through time. Option only available for RNNs. (default = 1)')
# NOTE: Currently we find all rnns by doing regex. If this changes to be true, add this argument.
parser.add_argument('-rnn', '--rnn', dest='rnn', action='store_true',
                    help='Wheter we have an rnn or not. (not needed if arch str contains \'rnn\')')
parser.add_argument('-bl', '--balance', dest='balance', action='store_true',
                    help='Balance dataset by sampling with replacement. Not applicable for RNNs. Forces shuffle to True in training set.')
#parser.add_argument('-t', '--timeout', default=60*6, type=int, dest='timeout',
#                    metavar='N', help='Maximum number of minutes to train. (default = 360)')

#TODO: data_to_dict, dataset, main.
# save, load,
args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'
PATH_RESUME = PATH_BASE + 'models/' + args.resume
PATH_SAVE = PATH_BASE + 'models/' + args.save_path
PATH_DATA = PATH_BASE + 'dataset/' + args.dataset_path
NUM_WORKERS = 3
PIN_MEM = False

if args.manual_past_frames:
    args.manual_past_frames = [int(i) for i in args.manual_past_frames.split(' ')]

rnn_arch_match = re.search('RNN', args.arch, flags=re.IGNORECASE)
if rnn_arch_match is not None:
    args.rnn = True

# find lr and momentum from optimizer settings
lr_match = re.search('(?<=lr=)\d*e-\d*', args.optim) #TODO remove space ' ' sensitivity
learning_rate = float(lr_match.group()) if lr_match is not None else 0
momentum_match = re.search('(?<=momentum=)\d*\.\d*', args.optim) #TODO remove space ' ' sensitivity
momentum = float(momentum_match.group()) if momentum_match is not None else 0
if args.scheduler and (learning_rate == 0 or momentum == 0):
    print("\n SCHEDULER WARNING: Could not find learning rate or momentum with regex. \
        Learning rate is " + str(learning_rate) + " and momentum is " + str(momentum) )

def main():
    #write info file
    if not os.path.exists(PATH_SAVE):
            os.makedirs(PATH_SAVE)
    write_info_file()
    # variables
    best_res = 1000000 #big number
    epoch_start = 0
    step_start = 0
    train_losses = ResultMeter()
    validation_losses = ResultMeter()
    times = ResultMeter()
    lr_scheduler = Scheduler('lr')
    momentum_scheduler = Scheduler('momentum')

    # load all time best
    print("-----Load all time best loss (for comparision)-----")
    all_time_best_res = 1000000 # big number
    all_time_best = load_checkpoint( PATH_SAVE + 'all_time_best.pt')
    if all_time_best is not None:
        all_time_best_res = all_time_best['best_res']
        print("avg loss @:",all_time_best_res)
        del all_time_best # Remove from GPU memory

    # Load datasets
    print("-----Loading datasets-----")
    if not args.evaluate:
        dataloader_train = get_data_loader(PATH_DATA + 'train/', shuffle=args.shuffle, balance=args.balance) # ca 232.500 frames in total
        dataloader_val = get_data_loader(PATH_DATA + 'validate/', max=4000, shuffle=False, balance=False) # ca 35.000 frames in total in 4 subsets (validation set has only one category)
    dataloader_test = get_data_loader(PATH_DATA + 'test/', max=-1, shuffle = False, balance=False)  # ca 66.200 frames in total in 8 subsets

    # create new model and lossfunctions and stuff
    if not args.resume:
        print("-----Creating network-----")
        model = eval(args.arch + "()")
        model.cuda()
        print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))

        # define loss function and optimizer
        print("-----Creating lossfunction and optimizer-----")
        loss_fn = torch.nn.MSELoss().cuda()
        optimizer = eval('torch.optim.' + args.optim)
        if args.scheduler:
            lr_scheduler.setValues(len(dataloader_train)*args.epochs, learning_rate/10, learning_rate)
            momentum_scheduler.setValues(len(dataloader_train)*args.epochs, momentum, momentum-0.1)

    #resume from checkpoint
    if args.resume:
        print("----Resume from checkpoint:-----")
        checkpoint = load_checkpoint(PATH_RESUME)
        if checkpoint is None:
            print("No file found. Exiting...")
            return
        # model
        print("\t Creating network")
        args.arch = checkpoint['arch']
        model = eval(args.arch + "()")
        model.cuda()
        print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))
        # loss function and optimizer
        print("\t Creating lossfunction and optimizer")
        args.optim = checkpoint['optim']
        loss_fn = torch.nn.MSELoss().cuda()
        optimizer = eval('torch.optim.' + args.optim)

        #load variables
        print("\t Loading variables")
        args.past_frames = checkpoint['past_frames']
        args.frame_stride = checkpoint['frame_stride']
        args.manual_past_frames = checkpoint['manual_past_frames'] if ('manual_past_frames' in checkpoint) else ''
        epoch_start = checkpoint['epoch']
        step_start = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        best_res = checkpoint['best_res']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.deserialize(checkpoint['lr_scheduler'])
        momentum_scheduler.deserialize(checkpoint['momentum_scheduler'])
        train_losses.deserialize(checkpoint['train_losses'])
        validation_losses.deserialize(checkpoint['validation_losses'])
        times.deserialize(checkpoint['times'])

        del checkpoint
        print("Loaded checkpoint sucessfully")

    # Train network
    if not args.evaluate:
        print("______TRAIN MODEL_______")
        main_loop(epoch_start, step_start, model, optimizer, lr_scheduler,
                    momentum_scheduler, loss_fn, train_losses, validation_losses,
                    times, dataloader_train, dataloader_val, best_res, all_time_best_res)

    # Final evaluation on test dataset
    #print("_____EVALUATE MODEL______")
    #test_loss = validate(model, dataloader_test, loss_fn, True)
    #print("Test loss: %f" %test_loss)

def get_data_loader(path, max=-1, shuffle=False, balance=False, sampler_max = None):
    if args.manual_past_frames is None:
        args.manual_past_frames = list(range(args.frame_stride,
                                             args.frame_stride*args.past_frames+1,
                                             args.frame_stride))

    # We need to load data differently depending on the architecture
    if args.rnn:
        args.manual_past_frames = []
        # TODO Remove max arg
        data = get_data(path, args.manual_past_frames, -1)

        dataset = RNNDataset(data, args.no_intention,
                                bptt=args.past_frames+1,
                                frame_stride=args.frame_stride)

        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEM,
                                   drop_last=True,
                                   sampler=sampler)

    else:
        data = get_data(path, args.manual_past_frames, max)

        if balance:
            # calculate weights for balancing categories
            array = data['category']
            category_count = dict([(category, len(array[numpy.where(array == category)])) for category in numpy.unique(array)])
            print(category_count)
            equal_weight = 1/len(category_count)

            # Here we can set the probability for each category to be included in a batch.
            # Do not necessarily have to sum to 1
            rel_weights = {'left': 1,
                            'right': 1,
                            'left_intention': 1,
                            'right_intention': 1,
                            'straight': 1,
                            'traffic_light': 1,
                            'other': 0
                            }
            weights = [(equal_weight*rel_weights[x])/(category_count[x]) for x in array]
            if sampler_max is None:
                sampler_max = len(array)

            sampler = WeightedRandomSampler(weights, len(array), replacement=True)
            args.shuffle = False

        else:
            sampler = None

        dataset = OurDataset(data, args.no_intention)
        return DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=args.shuffle,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEM,
                                   sampler=sampler)

def main_loop(epoch_start, step_start, model, optimizer, lr_scheduler,
                momentum_scheduler, loss_fn, train_losses,
                validation_losses, times, dataloader_train, dataloader_val,
                best_res, all_time_best_res):

    print("train network for a total of {diff} epochs."\
            " [{epochs}/{total_epochs}]".format( \
            diff = max(args.epochs-epoch_start,0),
            epochs = epoch_start, total_epochs = args.epochs))

    step = step_start
    for epoch in range(epoch_start, args.epochs):
        # Train for one epoch
        print('Epoch length:',len(dataloader_train), '(mini-batches)')

        for i, batch in enumerate(dataloader_train):
            batch_start_time = time.time()

            # Train model with current batch and save loss and duration
            train_loss = train(model, batch, loss_fn, optimizer)
            train_losses.update(epoch + 1, i + 1, step, train_loss)
            times.update(epoch + 1, i + 1, step, time.time() - batch_start_time)

            #Update learning rate and momentum
            if args.scheduler:
                lr_scheduler.update(optimizer, step)
                momentum_scheduler.update(optimizer, step)

            # Print statistics
            if step % args.print_freq == 0:
                print_statistics(train_losses, times, len(dataloader_train))

            # Evaluate on validation set and save checkpoint
            if step % args.plot_freq == 0:
                validation_loss = validate(model, dataloader_val, loss_fn)
                validation_losses.update(epoch + 1, i + 1, step, validation_loss)
                # Save losses to csv for plotting
                save_statistic(train_losses, PATH_SAVE + 'train_losses.csv')
                save_statistic(validation_losses, PATH_SAVE + 'validation_losses.csv')

                # Store best loss value
                is_best = validation_loss < best_res
                best_res = min(validation_loss, best_res)
                is_all_time_best = validation_loss < all_time_best_res
                all_time_best_res = min(validation_loss, all_time_best_res)

                # Save checkpoint
                save_checkpoint({
                    'arch' : args.arch,
                    'past_frames': args.past_frames,
                    'frame_stride': args.frame_stride,
                    'manual_past_frames': args.manual_past_frames,
                    'epoch': epoch + 1,
                    'step' : step + 1,
                    'state_dict': model.state_dict(),
                    'best_res': best_res,
                    'optim' : args.optim,
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.serialize(),
                    'momentum_scheduler' : momentum_scheduler.serialize(),
                    'train_losses' : train_losses.serialize(),
                    'validation_losses' : validation_losses.serialize(),
                    'times' : times.serialize()
                }, is_best, is_all_time_best)

            step += 1
            #end of dataloader loop

        #print epoch results
        print_statistics(train_losses, times, len(dataloader_train))

        validation_loss = validate(model, dataloader_val, loss_fn)
        validation_losses.update(epoch + 1, i + 1, step, validation_loss)

        # Store best loss value
        is_best = validation_loss < best_res
        best_res = min(validation_loss, best_res)
        is_all_time_best = validation_loss < all_time_best_res
        all_time_best_res = min(validation_loss, all_time_best_res)

        # Save checkpoint
        save_checkpoint({
            'arch' : args.arch,
            'past_frames': args.past_frames,
            'frame_stride': args.frame_stride,
            'manual_past_frames': args.manual_past_frames,
            'epoch': epoch + 1,
            'step' : step + 1,
            'state_dict': model.state_dict(),
            'best_res': best_res,
            'optim' : args.optim,
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.serialize(),
            'momentum_scheduler' : momentum_scheduler.serialize(),
            'train_losses' : train_losses.serialize(),
            'validation_losses' : validation_losses.serialize(),
            'times' : times.serialize()
        }, is_best, is_all_time_best)

        print('Validation complete. Final results: \n'
                'Loss {losses.val:.4f} ({losses.avg:.4f})'\
                .format(losses=validation_losses))

def print_statistics(losses, times, batch_length):
    print('Epoch: [{0}][{1}/{2}]\t'
          'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Batch loss {losses.val:.4f} ({losses.avg:.4f})'.format( losses.epoch,
           losses.batch, batch_length, batch_time=times, losses=losses))

def train(model, batch, loss_fn, optimizer):
    model.train() # switch to train mode

    lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))
    values = Variable((batch['value']).type(torch.cuda.FloatTensor))
    targets = Variable((batch['output']).type(torch.cuda.FloatTensor))

    output = model(lidars, values)
    loss = loss_fn(output, targets)

    optimizer.zero_grad() # reset gradients
    loss.backward()
    optimizer.step() # update weights

    return loss.data[0] # return loss for this batch

def validate(model, dataloader, loss_fn, save_output=False):
    model.eval() # switch to eval mode
    losses = ResultMeter()

    for i, batch in enumerate(dataloader):
        # Read input and output into variables
        lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))#,volatile=True)
        values = Variable((batch['value']).type(torch.cuda.FloatTensor))#,volatile=True)
        targets = Variable((batch['output']).type(torch.cuda.FloatTensor))#,volatile=True)
        indices = batch['indices']

        # Run model and calculate loss
        output = model(lidars, values)
        loss = loss_fn(output, targets)

        # Save generated predictions to file
        if save_output:
            current_batch_size = numpy.size(output,0)
            outputs = numpy.split(output.data, current_batch_size, 0)
            path = PATH_SAVE + 'generated_output/'
            generate_output(indices, outputs, batch['foldername'], path)

        # Document results
        losses.update(0, 0, i, loss.data[0])

        # Print statistics
        if i % args.print_freq == 0:
            print('Validation: [{batch}/{total}]\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format( batch = i,
                    total = len(dataloader), losses=losses))

    return losses.avg

def generate_output(indices, outputs, foldername, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, output in enumerate(outputs):
        subpath = path + str(foldername[i]) + '/'
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        output = output.view(-1,2)
        filename = subpath + 'gen_%i.csv' %(int(indices[i]))
        numpy.savetxt(filename, output, comments='', delimiter=',',fmt='%.8f',
                        header='x,y')

def save_checkpoint(state, is_best, is_all_time_best,
        filename = PATH_SAVE + 'checkpoint.pt'):
    torch.save(state, filename)
    #if is_best:
    #    print("Best so far!")
    #    shutil.copyfile(filename, PATH_SAVE + 'best.pt')
    if is_all_time_best:
        print("ALL TIME BEST! GOOD JOB!")
        shutil.copyfile(filename, PATH_SAVE + 'all_time_best.pt')

def load_checkpoint(filename):
    if os.path.isfile(filename):
        print("Loading model at '{}'".format(filename))
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print("No file found at '{}'".format(filename))
        return None

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def save_statistic(result_meter, path):
    values = numpy.array(result_meter.values)
    numpy.savetxt(path, values, comments='', delimiter=',',fmt='%.6f')

def write_info_file():
    if not args.evaluate:
        info = ""
        for key in args.__dict__:
            info += str(key) + " : " + str(args.__dict__[key]) + "\n"

        file = open(PATH_SAVE + "info.txt", "w")
        file.write(info)
        file.close()

if __name__ == '__main__':
    main()
