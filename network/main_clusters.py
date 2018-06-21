# -*- coding: utf-8 -*-
''' This is the main module for training networks which make use of clusters of
trajectories. The future path is predicted to be in one of several different
clusters. The centroid of the predicted cluster with highest probability is then
modified using delta values, which are also predicted. The delta values are
differences in principal component values and are added to the centroids'
principal components. The resulting trajectory is compared with the ground truth
using MSE.'''

import torch
import pandas
import numpy
import argparse
import time
import os
import re
import shutil
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
parser.add_argument('--only-lidar', dest='only_lidar', action='store_true',
                    help='Set all non-lidar values to 0. (default: False)). If False, --no-intention is still valid.')
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
parser.add_argument('-d','--dataset', dest='dataset_name', default='',
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

parser.add_argument('-ipf','--input_past_frames', default=None, type=str, metavar='\'1 2 3\'',
                    help = 'Which input past frames we want to use. (Instead of using 30 always) \
                    List which frames you want manually. Ex: \'1 3 5 7 10 13 16\''\
                    'NOTE: Do not inclue 0.')
parser.add_argument('-off','--output_future_frames', default=None, type=str, metavar='\'1 2 3\'',
                    help = 'Which output future frames we want to use as target. (Instead of using 30 always) \
                    List which frames you want manually. Ex: \'1 3 5 7 10 13 16\''\
                    'NOTE: Do not include 0.')
# NOTE not used at the moment
#parser.add_argument('-bptt', '--bptt', default=1, type=int, dest='bptt',
#                    metavar='N', help='Back propagation through time. Option only available for RNNs. (default = 1)')
# NOTE: Currently we find all rnns by doing regex. If this changes to be true, add this argument.
parser.add_argument('-rnn', '--rnn', dest='rnn', action='store_true',
                    help='Wheter we have an rnn or not. (not needed if arch str contains \'rnn\')')
parser.add_argument('-bl', '--balance', dest='balance', action='store_true',
                    help='Balance dataset by sampling with replacement. Not applicable for RNNs. Forces shuffle to True in training set.')
parser.add_argument('-t', '--timeout', default=None, type=int, dest='timeout',
                    metavar='N', help='Maximum number of minutes to train. After this time a validation is done. (default = None)')
parser.add_argument('-npc', '--n-components', default=10, type=int, dest='n_pc',
                    metavar='N', help='Number of leading principal components to be predicted. (default = 10)')
parser.add_argument('-ncl', '--n-clusters', default=10, type=int, dest='n_clusters',
                    metavar='N', help='Number of clusters to choose from. (default = 10)')
parser.add_argument('-cpath','--cluster-path', default='', type=str, metavar='PATH',
                    help='Full path to folder containing cluster data and principal component data.')
# TODO

args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'
PATH_RESUME = PATH_BASE + 'models/' + args.resume
PATH_SAVE = PATH_BASE + 'models/' + args.save_path
PATH_DATA = PATH_BASE + 'dataset/' + args.dataset_name
NUM_WORKERS = 3
PIN_MEM = False

if args.input_past_frames:
    args.input_past_frames = [int(i) for i in args.input_past_frames.split(' ')]

if args.output_future_frames:
    args.output_future_frames = [int(i) for i in args.output_future_frames.split(' ')]

if args.manual_past_frames:
    args.manual_past_frames = [int(i) for i in args.manual_past_frames.split(' ')]

if args.manual_past_frames is None:
    args.manual_past_frames = list(range(args.frame_stride,
                                         args.frame_stride*args.past_frames+1,
                                         args.frame_stride))

rnn_arch_match = re.search('RNN', args.arch, flags=re.IGNORECASE)
if rnn_arch_match is not None:
    args.rnn = True

# find lr and momentum from optimizer settings
#lr_match = re.search('(?<=lr=)\d*e-\d*', args.optim) #TODO remove space ' ' sensitivity
lr_match = re.search('(?<=lr=)(.*?)(?=,)', args.optim) # NOTE Fulhack för att klara av t.ex. '3*1e-5'
learning_rate = float(eval(lr_match.group())) if lr_match is not None else 0
momentum_match = re.search('(?<=momentum=)\d*\.\d*', args.optim) #TODO remove space ' ' sensitivity
momentum = float(momentum_match.group()) if momentum_match is not None else 0
if args.scheduler and (learning_rate == 0 or momentum == 0):
    print("\n SCHEDULER WARNING: Could not find learning rate or momentum with regex. \
        Learning rate is " + str(learning_rate) + " and momentum is " + str(momentum) )

# start matlab engine
#eng = matlab.engine.start_matlab()
#matlab_wd = '/home/annaochjacob/Repos/exjobb/preprocessing/path_clustering/'
#eng.addpath(matlab_wd)
#eng.cd(matlab_wd)

# Load centroid

def main():

    #write info file
    if not os.path.exists(PATH_SAVE):
            os.makedirs(PATH_SAVE)
    write_info_file()

    # variables
    best_res = 1000000 #big number
    epoch_start = 0
    step_start = 0
    train_class_losses = ResultMeter()
    validation_class_losses = ResultMeter()
    train_regression_losses = ResultMeter()
    validation_regression_losses = ResultMeter()
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
        dataloader_train = get_data_loader(PATH_DATA + 'train/', shuffle=args.shuffle, balance=args.balance)
        dataloader_val = get_data_loader(PATH_DATA + 'validate/', shuffle=False, balance=False)
    dataloader_test = get_data_loader(PATH_DATA + 'test/', shuffle = False, balance=False)

    # create new model and lossfunctions and stuff
    if not args.resume:
        print("-----Creating network-----")
        n_ipf = str(len(args.input_past_frames))
        n_off = str(len(args.output_future_frames))
        n_mpf = str(len(args.manual_past_frames))
        model_arg_string = n_mpf + ', ' + n_ipf + ', ' + n_off + ', ' + str(args.n_pc) + ', ' + str(args.n_clusters)
        model = eval(args.arch + '(' + model_arg_string + ')')
        model.cuda()
        print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))

        # define loss function and optimizer
        print("-----Creating lossfunction and optimizer-----")
        trajectory_loss_fn = torch.nn.MSELoss().cuda()
        class_loss_fn = torch.nn.CrossEntropyLoss().cuda()
        optimizer = eval('torch.optim.' + args.optim)
        if args.scheduler:
            lr_scheduler.setValues(len(dataloader_train)*args.epochs, learning_rate/10, learning_rate)
            momentum_scheduler.setValues(len(dataloader_train)*args.epochs, momentum+0.05, momentum-0.05)
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
        args.past_frames = checkpoint['past_frames']
        args.frame_stride = checkpoint['frame_stride']
        args.manual_past_frames = checkpoint['manual_past_frames']
        args.input_past_frames = checkpoint['input_past_frames']
        args.output_future_frames = checkpoint['output_future_frames']
        n_ipf = str(len(args.input_past_frames))
        n_off = str(len(args.output_future_frames))
        n_mpf = str(len(args.manual_past_frames))
        model_arg_string = n_mpf + ', ' + n_ipf + ', ' + n_off + ', ' + str(n_pc) + ', ' + str(n_clusters)
        model = eval(args.arch + '(' + model_arg_string + ')')
        model.cuda()
        print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))
        # loss function and optimizer
        print("\t Creating lossfunction and optimizer")
        args.optim = checkpoint['optim']
        loss_fn = torch.nn.MSELoss().cuda()
        optimizer = eval('torch.optim.' + args.optim)

        #load variables
        print("\t Loading variables")
        epoch_start = checkpoint['epoch']
        step_start = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        best_res = checkpoint['best_res']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.deserialize(checkpoint['lr_scheduler'])
        momentum_scheduler.deserialize(checkpoint['momentum_scheduler'])
        train_class_losses.deserialize(checkpoint['train_class_losses'])
        validation_class_losses.deserialize(checkpoint['validation_class_losses'])
        train_regression_losses.deserialize(checkpoint['train_regression_losses'])
        validation_regression_losses.deserialize(checkpoint['validation_regression_losses'])
        times.deserialize(checkpoint['times'])

        del checkpoint
        print("Loaded checkpoint sucessfully")

    # Train network
    if not args.evaluate:
        print("______TRAIN MODEL_______")
        main_loop(epoch_start, step_start, model, optimizer, lr_scheduler,
                    momentum_scheduler, class_loss_fn, trajectory_loss_fn,
                    train_class_losses, validation_class_losses,
                    train_regression_losses, validation_regression_losses,
                    times, dataloader_train, dataloader_val, best_res,
                    all_time_best_res, args.n_pc, args.n_clusters,
                    args.cluster_path, args.timeout)

    # Final evaluation on test dataset
    if args.evaluate:
        print("_____EVALUATE MODEL______")
        test_class_loss, test_regression_loss = validate(model, dataloader_test,
                            class_loss_fn, trajectory_loss_fn, coeffs, means,
                            centroids, optimizer, n_pc, n_clusters, True)
        print("Test regression loss: %f" %test_regression_loss)
        print("Test classification loss: %f" %test_class_loss)

def get_data_loader(path, shuffle=False, balance=False, sampler_max = None):
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

    # NOT RNN
    data = get_data(path, args.manual_past_frames)
    dataset = OurDataset(path, data, args.no_intention, args.only_lidar,
                args.manual_past_frames, args.input_past_frames, args.output_future_frames)
    categories = numpy.array(dataset.categories)
    weights = [1]*len(categories)
    replacement = False

    # calculate weights for balancing categories
    if balance:
        category_count = dict([(category, len(categories[numpy.where(categories == category)])) for category in numpy.unique(categories)])
        equal_weight = 1/len(category_count)
        # Here we can set the probability for each category to be included in a batch.
        # Do not necessarily have to sum to 1
        rel_weights = {0: 1, # straight
                        1: 1, # right intention
                        2: 1, # left intention
                        3: 1, # right
                        4: 1, # left
                        5: 0, # other
                        6: 1  # traffic light
                        }
        weights = [(equal_weight*rel_weights[x])/(category_count[x]) for x in categories]
        replacement = True

    if sampler_max is None: sampler_max = len(weights)
    sampler = WeightedRandomSampler(weights, sampler_max, replacement=replacement)

    return DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=args.shuffle,
                               num_workers=NUM_WORKERS,
                               pin_memory=PIN_MEM,
                               sampler=sampler)

def main_loop(epoch_start, step_start, model, optimizer, lr_scheduler,
                momentum_scheduler,  class_loss_fn, trajectory_loss_fn,
                train_class_losses, validation_class_losses,
                train_regression_losses, validation_regression_losses,
                times, dataloader_train, dataloader_val,
                best_res, all_time_best_res, n_pc, n_clusters, cluster_path, timeout=None):

    # Load principal component coefficient (or factor loadings) matrix
    coeffs = numpy.genfromtxt(cluster_path + 'coeff.csv',delimiter=',')

    # Use only the n_pc first components
    coeffs = coeffs[:,0:n_pc]
    coeffs = Variable(torch.cuda.FloatTensor(coeffs), requires_grad=False)

    # Load mean of variables for PCA reconstruction
    means = numpy.genfromtxt(cluster_path + 'variable_mean.csv',delimiter=',')
    means = Variable(torch.cuda.FloatTensor(means), requires_grad=False)

    # Load cluster centroids
    centroids = numpy.genfromtxt(cluster_path + 'centroids.csv',delimiter=',')
    centroids = Variable(torch.cuda.FloatTensor(centroids), requires_grad=False)
    #n_clusters = centroids.shape[0]

    # Load cluster indices for trajectories (used for debugging)
    cluster_idx = numpy.genfromtxt(cluster_path + 'cluster_idx.csv',delimiter=',')

    # Make zero-indexed for convenience
    cluster_idx = cluster_idx - 1

    print("train network for a total of {diff} epochs."\
            " [{epochs}/{total_epochs}]".format( \
            diff = max(args.epochs-epoch_start,0),
            epochs = epoch_start, total_epochs = args.epochs))

    training_start_time = time.time()
    time_is_out = False
    step = step_start
    for epoch in range(epoch_start, args.epochs):
        # Train for one epoch
        print('Epoch length:',len(dataloader_train), '(mini-batches)')

        for i, batch in enumerate(dataloader_train):


            batch_start_time = time.time()

            # Train model with current batch and save loss and duration
            train_class_loss, train_regression_loss = train(model, batch,
                class_loss_fn, trajectory_loss_fn, coeffs,means, centroids,
                optimizer, n_pc, n_clusters)
            train_class_losses.update(epoch + 1, i + 1, step, train_class_loss)
            train_regression_losses.update(epoch + 1, i + 1, step, train_regression_loss)
            times.update(epoch + 1, i + 1, step, time.time() - batch_start_time)

            #Update learning rate and momentum
            if args.scheduler:
                lr_scheduler.update(optimizer, step)
                momentum_scheduler.update(optimizer, step)

            # Print statistics
            if step % args.print_freq == 0:
                print_statistics(train_regression_losses, times, len(dataloader_train))

            # Check for timeout and possibly break
            if timeout is not None:
                minutes_trained = (batch_start_time - training_start_time)/60
                time_is_out = minutes_trained > timeout
                if time_is_out:
                    print('Training reached timeout at %.2f min' %minutes_trained)
                    break

            # Evaluate on validation set and save checkpoint
            if step % args.plot_freq == 0 and step != 0:
                validation_class_loss, validation_regression_loss = validate(model,
                    dataloader_val, class_loss_fn, trajectory_loss_fn, coeffs,
                    means, centroids, optimizer, n_pc, n_clusters)

                validation_class_losses.update(epoch + 1, i + 1, step, validation_class_loss)
                validation_regression_losses.update(epoch + 1, i + 1, step, validation_regression_loss)
                # Save losses to csv for plotting
                save_statistic(train_class_losses, PATH_SAVE + 'train_class_losses.csv')
                save_statistic(validation_class_losses, PATH_SAVE + 'validation_class_losses.csv')
                save_statistic(train_regression_losses, PATH_SAVE + 'train_regression_losses.csv')
                save_statistic(validation_regression_losses, PATH_SAVE + 'validation_regression_losses.csv')

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
                    'input_past_frames' : args.input_past_frames,
                    'output_future_frames': args.output_future_frames,
                    'epoch': epoch + 1,
                    'step' : step + 1,
                    'state_dict': model.state_dict(),
                    'best_res': best_res,
                    'optim' : args.optim,
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.serialize(),
                    'momentum_scheduler' : momentum_scheduler.serialize(),
                    'train_class_losses' : train_class_losses.serialize(),
                    'validation_class_losses' : validation_class_losses.serialize(),
                    'train_regression_losses' : train_regression_losses.serialize(),
                    'validation_regression_losses' : validation_regression_losses.serialize(),
                    'times' : times.serialize()
                }, is_best, is_all_time_best)

            step += 1
            #end of dataloader loop

        #print epoch results
        print_statistics(train_regression_losses, times, len(dataloader_train))

        validation_class_loss, validation_regression_loss = validate(model,
            dataloader_val, class_loss_fn, trajectory_loss_fn, coeffs, means,
            centroids, optimizer, n_pc, n_clusters)

        validation_class_losses.update(epoch + 1, i + 1, step, validation_class_loss)
        validation_regression_losses.update(epoch + 1, i + 1, step, validation_regression_loss)
        # Save losses to csv for plotting
        save_statistic(train_class_losses, PATH_SAVE + 'train_class_losses.csv')
        save_statistic(validation_class_losses, PATH_SAVE + 'validation_class_losses.csv')
        save_statistic(train_regression_losses, PATH_SAVE + 'train_regression_losses.csv')
        save_statistic(validation_regression_losses, PATH_SAVE + 'validation_regression_losses.csv')

        # Store best loss value
        is_best = validation_regression_loss < best_res
        best_res = min(validation_regression_loss, best_res)
        is_all_time_best = validation_regression_loss < all_time_best_res
        all_time_best_res = min(validation_regression_loss, all_time_best_res)

        # Save checkpoint
        save_checkpoint({
            'arch' : args.arch,
            'past_frames': args.past_frames,
            'frame_stride': args.frame_stride,
            'manual_past_frames': args.manual_past_frames,
            'input_past_frames' : args.input_past_frames,
            'output_future_frames': args.output_future_frames,
            'epoch': epoch + 1,
            'step' : step + 1,
            'state_dict': model.state_dict(),
            'best_res': best_res,
            'optim' : args.optim,
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.serialize(),
            'momentum_scheduler' : momentum_scheduler.serialize(),
            'train_class_losses' : train_class_losses.serialize(),
            'validation_class_losses' : validation_class_losses.serialize(),
            'train_regression_losses' : train_regression_losses.serialize(),
            'validation_regression_losses' : validation_regression_losses.serialize(),
            'times' : times.serialize()
        }, is_best, is_all_time_best)

        print('Validation complete. Final results: \n' \
                'Regression loss {reg_losses.val:.4f} ({reg_losses.avg:.4f})\n'\
                'Classification loss {class_losses.val:.4f} ({class_losses.avg:.4f})\n'\
                .format(reg_losses=validation_regression_losses,
                        class_losses=validation_class_losses))

        if time_is_out: return # Do not continue with next epoch if time is out

def print_statistics(losses, times, batch_length):
    print('Epoch: [{0}][{1}/{2}]\t'
          'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Batch loss {losses.val:.4f} ({losses.avg:.4f})'.format( losses.epoch,
           losses.batch, batch_length, batch_time=times, losses=losses))

def train(model, batch, class_loss_fn, trajectory_loss_fn, coeffs, means,
            centroids, optimizer, n_pc, n_clusters):
    model.train() # switch to train mode

    # These are the lidar and value matrices used as input
    lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))
    values = Variable((batch['value']).type(torch.cuda.FloatTensor))

    # This is the ground truth trajectory
    targets = Variable((batch['output']).type(torch.cuda.FloatTensor))

    # Get batch size for creating tensor views later
    batch_size = targets.shape[0]

    # Find the cluster index for each of the ground truth trajectories in batch
    # NOTE target_class has been verified to work as expected
    target_class = get_cluster_idx(targets, centroids)
    target_class = Variable(target_class.cuda(), requires_grad=False)

    # Compute principal component adjustments and predict the trajectory class
    pc_deltas, class_logits = model(lidars,values)

    # Make view of pc_delta
    # from shape [batch_size, n_clusters * n_pc]
    # to shape [batch_size, n_clusters, n_pc]
    # NOTE view operation has been verified to work as expected
    pc_deltas = pc_deltas.view(batch_size, n_clusters, n_pc)

    # Get index of predicted class with highest value
    #(not yet on the form of a probability distribution)
    # NOTE max operation has been verified to work as expected
    _, predicted_class = torch.max(class_logits,1)

    # Compute loss on the class prediction. Class_loss then is a scalar tensor
    # containing the average loss over observations in the minibatch.
    class_loss = class_loss_fn(class_logits, target_class)

    # Add deltas to predicted cluster centroid principal components
    # NOTE everything from here up to and including pc_adjusted is verified to work as expected
    predicted_centroid = centroids[predicted_class]
    pc_centroid = torch.mm(predicted_centroid, coeffs) # [batch_size, 60] * [60, n_pc] = [batch_size, n_pc]
    pc_deltas_predicted = pc_deltas[numpy.arange(0,batch_size),predicted_class]
    pc_adjusted = torch.add(pc_centroid, pc_deltas_predicted)

    # Go from principal components to original projection
    # NOTE predicted_trajectory is verified to work as expected
    predicted_trajectory = torch.mm(pc_adjusted,coeffs.transpose(0,1)) + means

    # Compute loss on the adjusted trajectory
    trajectory_loss = trajectory_loss_fn(predicted_trajectory, targets)

    # Compute the total loss as the sum of class loss and trajectory loss
    total_loss = class_loss + trajectory_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return class_loss.data, trajectory_loss.data

def validate(model, dataloader, class_loss_fn, trajectory_loss_fn, coeffs, means,
            centroids, optimizer, n_pc, n_clusters, save_output=False):
    model.eval() # switch to eval mode
    regression_losses = ResultMeter() # Record trajectory MSE loss
    class_losses = ResultMeter() # Record classification Crossentropy losses

    for i, batch in enumerate(dataloader):
        # Read input and output into variables
        lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))#,volatile=True)
        values = Variable((batch['value']).type(torch.cuda.FloatTensor))#,volatile=True)
        targets = Variable((batch['output']).type(torch.cuda.FloatTensor))#,volatile=True)
        indices = batch['index']

        # Get batch size for creating tensor views later
        batch_size = targets.shape[0]

        # Find the cluster index for each of the ground truth trajectories in batch
        # NOTE target_class has been verified to work as expected
        target_class = get_cluster_idx(targets, centroids)
        target_class = Variable(target_class.cuda(), requires_grad=False)

        # Compute principal component adjustments and predict the trajectory class
        pc_deltas, class_logits = model(lidars,values)

        # Make view of pc_delta
        # from shape [batch_size, n_clusters * n_pc]
        # to shape [batch_size, n_clusters, n_pc]
        # NOTE view operation has been verified to work as expected
        pc_deltas = pc_deltas.view(batch_size, n_clusters, n_pc)

        # Get index of predicted class with highest value
        #(not yet on the form of a probability distribution)
        # NOTE max operation has been verified to work as expected
        _, predicted_class = torch.max(class_logits,1)

        # Compute loss on the class prediction. Class_loss then is a scalar tensor
        # containing the average loss over observations in the minibatch.
        class_loss = class_loss_fn(class_logits, target_class)

        # Add deltas to predicted cluster centroid principal components
        # NOTE everything from here up to and including pc_adjusted is verified to work as expected
        predicted_centroid = centroids[predicted_class]
        pc_centroid = torch.mm(predicted_centroid, coeffs) # [batch_size, 60] * [60, n_pc] = [batch_size, n_pc]
        pc_deltas_predicted = pc_deltas[numpy.arange(0,batch_size),predicted_class]
        pc_adjusted = torch.add(pc_centroid, pc_deltas_predicted)

        # Go from principal components to original projection
        # NOTE predicted_trajectory is verified to work as expected
        predicted_trajectory = torch.mm(pc_adjusted,coeffs.transpose(0,1)) + means

        # Compute loss on the adjusted trajectory
        trajectory_loss = trajectory_loss_fn(predicted_trajectory, targets)

        # Compute the total loss as the sum of class loss and trajectory loss
        total_loss = class_loss + trajectory_loss

        # Save generated predictions to file
        if save_output:
            current_batch_size = numpy.size(output,0)
            outputs = numpy.split(output.data, current_batch_size, 0)
            path = PATH_SAVE + 'generated_output/'
            generate_output(indices, outputs, batch['foldername'], path)

        # Document results
        regression_losses.update(0, 0, i, trajectory_loss.data[0])
        class_losses.update(0, 0, i, class_loss.data[0])

        # Print statistics
        if i % args.print_freq == 0:
            print('Validation: [{batch}/{total}]\t'
                  'Regression Loss {regression_losses.val:.4f}({regression_losses.avg:.4f})'.format( batch = i,
                    total = len(dataloader), regression_losses=regression_losses))

    return class_losses.avg, regression_losses.avg

def generate_output(indices, outputs, foldername, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, output in enumerate(outputs):
        #print(foldername[i])
        subpath = path + str(foldername[i]) + '/'
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        output = output.view(2,-1).transpose(0,1)
        filename = subpath + 'gen_%i.csv' %(int(indices[i]))
        numpy.savetxt(filename, output, comments='', delimiter=',',fmt='%.8f',
                        header='x,y')

def save_checkpoint(state, is_best, is_all_time_best,
                    filename = PATH_SAVE + 'checkpoint.pt'):
    torch.save(state, filename)

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

def get_cluster_idx(points, centroids):
    # Use the squared euclidean distance, since that was used when
    # constructing cluster centroids.
    indices = []
    for point in points:
        # Calculate distance from point to each centroid
        distances = torch.FloatTensor([(point - c).pow(2).sum() for c in centroids])
        cluster_idx = torch.argmin(distances)
        indices.append(cluster_idx)

    return torch.LongTensor(indices)

if __name__ == '__main__':
    main()
