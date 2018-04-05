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

from architectures import *
from data_to_dict import getData
from dataset import OurDataset
#from architectures.network import LucaNetwork, SmallerNetwork1, SmallerNetwork2
from result_meter import ResultMeter

# NOTE! Validate during traning. Test is last when model finished traning.

parser = argparse.ArgumentParser(description='PyTorch Drive a car wohoo')
parser.add_argument('-a','--arch', default='', type=str, metavar='file.class',
                    help = 'Name of network to use. eg: LucaNetwork.LucaNet')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
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
parser.add_argument('-d','--dataset', dest='dataset_path', default='temp/',
                    type=str, metavar='PATH',
                    help = 'Name of folder in /media/annaochjacob/crucial/dataset/ ex \'Banana_split/\' (with trailing /)')
parser.add_argument('-s','--save-path', dest='save_path', default='temp/',
                    type=str, metavar='PATH',
                    help = 'Name of folder in /media/annaochjacob/crucial/models/ ex \'SmallerNetwork1/\' (with trailing /)')

args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'
PATH_RESUME = PATH_BASE + 'models/' + args.resume
PATH_SAVE = PATH_BASE + 'models/' + args.save_path
PATH_DATA = PATH_BASE + 'dataset/' + args.dataset_path

def main():
    # variables
    best_res = 10000 #big number
    learning_rate = 1e-5 #1e-4 NOTE too large can give NaN in forward pass! (Use 1e-4 and lower)
    epoch_start = 0
    step_start = 0
    train_losses = ResultMeter()
    validation_losses = ResultMeter()
    times = ResultMeter()

    # load all time best
    print("-----Load all time best loss (for comparision)-----")
    all_time_best_res = 10000 # big number
    all_time_best = load_checkpoint( PATH_SAVE + 'all_time_best.pt')
    if all_time_best is not None:
        all_time_best_res = all_time_best['best_res']
        print("avg loss @:",all_time_best_res)
        del all_time_best # Remove from GPU memory

    # create new model and lossfunctions and stuff
    if not args.resume:
        print("-----Creating network-----")
        model = eval(args.arch + "()")
        model.cuda()
        print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))

        # define loss function and optimizer
        print("-----Creating lossfunction and optimizer-----")
        loss_fn = torch.nn.MSELoss().cuda()
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

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
        loss_fn = torch.nn.MSELoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

        #load variables
        print("\t Loading variables")
        epoch_start = checkpoint['epoch']
        step_start = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        best_res = checkpoint['best_res']
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_losses.deserialize(checkpoint['train_losses'])
        validation_losses.deserialize(checkpoint['validation_losses'])
        times.deserialize(checkpoint['times'])

        del checkpoint
        print("Loaded checkpoint sucessfully")

    # Load datasets
    print("-----Loading datasets-----")
    # TODO if evaluate flag is true, don't load all the datasets.
    if not args.evaluate:
        validate_dataset = OurDataset(getData(PATH_DATA + 'validate/')) #2000
        train_dataset = OurDataset(getData(PATH_DATA + 'train/')) #14000
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=False)
        dataloader_val = DataLoader(validate_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=False)

    test_dataset = OurDataset(getData(PATH_DATA + 'test/')) #4000
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=False)

    # Train network
    if not args.evaluate:
        print("______TRAIN MODEL_______")
        if not os.path.exists(PATH_SAVE):
                os.makedirs(PATH_SAVE)
        main_loop(epoch_start, step_start, model, optimizer, loss_fn, train_losses,
                    validation_losses, times, dataloader_train, dataloader_val,
                    best_res, all_time_best_res)

    # Final evaluation on test dataset
    print("_____EVALUATE MODEL______")
    test_loss = validate(model, dataloader_test, loss_fn, True)
    print("Test loss: %f" %test_loss)

def main_loop(epoch_start, step_start, model, optimizer, loss_fn, train_losses,
                validation_losses, times, dataloader_train, dataloader_val,
                best_res, all_time_best_res):

    print("train network for a total of {diff} epochs."\
            " [{epochs}/{total_epochs}]".format( \
            diff = max(args.epochs-epoch_start,0),
            epochs = epoch_start, total_epochs = args.epochs))

    step = step_start
    for epoch in range(epoch_start, args.epochs):
        # Train for one epoch
        for i, batch in enumerate(dataloader_train):
            start_time = time.time()

            # Train model with current batch and save loss and duration
            train_loss = train(model, batch, loss_fn, optimizer)
            train_losses.update(epoch + 1, i + 1, step, train_loss)
            times.update(epoch + 1, i + 1, step, time.time() - start_time)

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
            'epoch': epoch + 1,
            'step' : step + 1,
            'state_dict': model.state_dict(),
            'best_res': best_res,
            'optimizer' : optimizer.state_dict(),
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
          'Loss {losses.val:.4f} ({losses.avg:.4f})'.format( losses.epoch,
           losses.batch, batch_length, batch_time=times, losses=losses))

def train(model, batch, loss_fn, optimizer):
    model.train() # switch to train mode

    lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))
    values = Variable((batch['value']).type(torch.cuda.FloatTensor))
    targets = Variable((batch['output']).type(torch.cuda.FloatTensor))

    lidars = lidars.view(-1, 1, 600, 600)
    values = values.view(-1, 30, 11)
    targets = targets.view(-1, 60)

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
        lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor),volatile=True)
        values = Variable((batch['value']).type(torch.cuda.FloatTensor),volatile=True)
        targets = Variable((batch['output']).type(torch.cuda.FloatTensor),volatile=True)
        indices = batch['indices']

        # Set correct shape on input and output
        lidars = lidars.view(-1, 1, 600, 600)
        values = values.view(-1, 30, 11)
        targets = targets.view(-1, 60)

        # Run model and calculate loss
        output = model(lidars, values)
        loss = loss_fn(output, targets)

        # Save generated predictions to file
        if save_output:
            current_batch_size = numpy.size(output,0)
            outputs = numpy.split(output.data, current_batch_size, 0)
            path = PATH_SAVE + 'generated_output/'
            generate_output(indices, outputs, i, path)

        # Document results
        losses.update(0, 0, i, loss.data[0])

    return losses.avg

def generate_output(indices, outputs, batch_index, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i, output in enumerate(outputs):
        output = output.view(-1,2)
        filename = path + 'gen_%i.csv' %(indices[i])
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
    #print("\n")

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


if __name__ == '__main__':
    main()
