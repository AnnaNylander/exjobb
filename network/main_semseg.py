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
from skimage import draw
from scipy import ndimage
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, WeightedRandomSampler

from architectures import *
from data_to_dict import get_data
from dataset import OurDataset, RNNDataset
from scheduler import Scheduler
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
                    metavar='N', help='Maximum number of minutes to train. After this time a validation is done. (default = None.')

parser.add_argument('-res', '--resolution', nargs=2, default=[300,300], type=int, dest='resolution',
                    metavar=('N','N'), help='Resolution of the output, e.g. 300x300 pixels. (default= 300 300)')
parser.add_argument('-r', '--radius', default=None, type=float, dest='radius',
                    metavar='N', help='The radius in meters of the target path circles for semantic segmentation idea (default=5)')


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
    MSE_train_losses = ResultMeter()
    validation_losses = ResultMeter()
    MSE_validation_losses = ResultMeter()
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
        dataloader_train = get_data_loader(PATH_DATA + 'train/', shuffle=args.shuffle, balance=args.balance, sampler_max= None)
        dataloader_val = get_data_loader(PATH_DATA + 'validate/', shuffle=False, balance=False, sampler_max= 1000)
    dataloader_test = get_data_loader(PATH_DATA + 'test/', shuffle = False, balance=False, sampler_max= None)

    # create new model and lossfunctions and stuff
    if not args.resume:
        print("-----Creating network-----")
        n_ipf = str(len(args.input_past_frames))
        n_off = str(len(args.output_future_frames))
        n_mpf = str(len(args.manual_past_frames))
        model_arg_string = n_mpf + ', ' + n_ipf + ', ' + n_off
        model = eval(args.arch + '(' + model_arg_string + ')')
        model.cuda()
        print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))

        # define loss function and optimizer
        print("-----Creating lossfunction and optimizer-----")
        loss_fn = torch.nn.BCELoss().cuda()
        MSE_loss_fn = torch.nn.MSELoss().cuda()
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
        args.resolution = checkpoint['resolution']
        if args.radius is None:
            args.radius = checkpoint['radius']
        args.past_frames = checkpoint['past_frames']
        args.frame_stride = checkpoint['frame_stride']
        args.manual_past_frames = checkpoint['manual_past_frames']
        args.input_past_frames = checkpoint['input_past_frames']
        args.output_future_frames = checkpoint['output_future_frames']
        n_ipf = str(len(args.input_past_frames))
        n_off = str(len(args.output_future_frames))
        n_mpf = str(len(args.manual_past_frames))
        model_arg_string = n_mpf + ', ' + n_ipf + ', ' + n_off
        model = eval(args.arch + '(' + model_arg_string + ')')
        model.cuda()
        print('Model size: %iMB' %(2*get_n_params(model)*4/(1024**2)))
        # loss function and optimizer
        print("\t Creating lossfunction and optimizer")
        args.optim = checkpoint['optim']
        loss_fn = torch.nn.BCELoss().cuda()
        MSE_loss_fn = torch.nn.MSELoss().cuda()
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
        train_losses.deserialize(checkpoint['train_losses'])
        validation_losses.deserialize(checkpoint['validation_losses'])
        MSE_train_losses.deserialize(checkpoint['MSE_train_losses'])
        MSE_validation_losses.deserialize(checkpoint['MSE_validation_losses'])
        times.deserialize(checkpoint['times'])

        del checkpoint
        print("Loaded checkpoint sucessfully")

    # Train network
    if not args.evaluate:
        print("______TRAIN MODEL_______")
        main_loop(epoch_start, step_start, model, optimizer, lr_scheduler,
                    momentum_scheduler, loss_fn, MSE_loss_fn, train_losses, MSE_train_losses,
                    validation_losses, MSE_validation_losses,
                    times, dataloader_train, dataloader_val, best_res,
                    all_time_best_res, args.timeout)

    # Final evaluation on test dataset
    if args.evaluate:
        print("_____EVALUATE MODEL______")
        test_class_loss, test_MSE_loss = validate(model, dataloader_test, loss_fn, MSE_loss_fn, True)
        print("Test loss: %f" %test_MSE_loss)
        write_test_loss_file(test_class_loss, test_MSE_loss)

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
    dataset = OurDataset(path, data, args.no_intention, args.only_lidar, args.manual_past_frames, args.input_past_frames, args.output_future_frames)
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
                momentum_scheduler, loss_fn, MSE_loss_fn, train_losses, MSE_train_losses,
                validation_losses, MSE_validation_losses, times, dataloader_train, dataloader_val,
                best_res, all_time_best_res, timeout=None):

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
            train_loss,MSE_train_loss = train(model, batch, loss_fn, MSE_loss_fn, optimizer)
            train_losses.update(epoch + 1, i + 1, step, train_loss)
            MSE_train_losses.update(epoch + 1, i + 1, step, MSE_train_loss)
            times.update(epoch + 1, i + 1, step, time.time() - batch_start_time)

            #Update learning rate and momentum
            if args.scheduler:
                lr_scheduler.update(optimizer, step)
                momentum_scheduler.update(optimizer, step)

            # Print statistics
            if step % args.print_freq == 0:
                print_statistics(train_losses, MSE_train_losses, times, len(dataloader_train))

            # Check for timeout and possibly break
            if timeout is not None:
                minutes_trained = (batch_start_time - training_start_time)/60
                time_is_out = minutes_trained > timeout
                if time_is_out:
                    print('Training reached timeout at %.2f min' %minutes_trained)
                    break

            # Evaluate on validation set and save checkpoint
            if step % args.plot_freq == 0 and step != 0:
                validation_loss,MSE_validation_loss = validate(model, dataloader_val, loss_fn, MSE_loss_fn)
                validation_losses.update(epoch + 1, i + 1, step, validation_loss)
                MSE_validation_losses.update(epoch + 1, i + 1, step, MSE_validation_loss)
                # Save losses to csv for plotting
                save_statistic(train_losses, PATH_SAVE + 'train_losses.csv')
                save_statistic(MSE_train_losses, PATH_SAVE + 'MSE_train_losses.csv')
                save_statistic(validation_losses, PATH_SAVE + 'validation_losses.csv')
                save_statistic(MSE_validation_losses, PATH_SAVE + 'MSE_validation_losses.csv')

                # Store best loss value
                is_best = MSE_validation_loss < best_res
                best_res = min(MSE_validation_loss, best_res)
                is_all_time_best = MSE_validation_loss < all_time_best_res
                all_time_best_res = min(MSE_validation_loss, all_time_best_res)

                # Save checkpoint
                save_checkpoint({
                    'arch' : args.arch,
                    'resolution': args.resolution,
                    'radius' : args.radius,
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
                    'train_losses' : train_losses.serialize(),
                    'validation_losses' : validation_losses.serialize(),
                    'MSE_train_losses' : MSE_train_losses.serialize(),
                    'MSE_validation_losses' : MSE_validation_losses.serialize(),
                    'times' : times.serialize()
                }, is_best, is_all_time_best)

            step += 1
            #end of dataloader loop

        #print epoch results
        print_statistics(train_losses, MSE_train_losses, times, len(dataloader_train))

        validation_loss, MSE_validation_loss = validate(model, dataloader_val, loss_fn, MSE_loss_fn)
        validation_losses.update(epoch + 1, i + 1, step, validation_loss)
        MSE_validation_losses.update(epoch + 1, i + 1, step, MSE_validation_loss)
        # Save losses to csv for plotting
        save_statistic(train_losses, PATH_SAVE + 'train_losses.csv')
        save_statistic(MSE_train_losses, PATH_SAVE + 'MSE_train_losses.csv')
        save_statistic(validation_losses, PATH_SAVE + 'validation_losses.csv')
        save_statistic(MSE_validation_losses, PATH_SAVE + 'MSE_validation_losses.csv')

        # Store best loss value
        is_best = MSE_validation_loss < best_res
        best_res = min(MSE_validation_loss, best_res)
        is_all_time_best = MSE_validation_loss < all_time_best_res
        all_time_best_res = min(MSE_validation_loss, all_time_best_res)

        # Save checkpoint
        save_checkpoint({
            'arch' : args.arch,
            'resolution': args.resolution,
            'radius' : args.radius,
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
            'train_losses' : train_losses.serialize(),
            'validation_losses' : validation_losses.serialize(),
            'MSE_train_losses' : MSE_train_losses.serialize(),
            'MSE_validation_losses' : MSE_validation_losses.serialize(),
            'times' : times.serialize()
        }, is_best, is_all_time_best)


        print('Validation complete. Final results: \n'
                'Loss {losses.val:.4f} ({losses.avg:.4f})\n'
                'MSE_Loss {MSE_losses.val:.4f} ({MSE_losses.avg:.4f})'\
                .format(losses=validation_losses, MSE_losses=MSE_validation_losses))

        if time_is_out: return # Do not continue with next epoch if time is out

def print_statistics(losses, MSE_losses, times, batch_length):
    print('Epoch: [{0}][{1}/{2}]\t'
          'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Batch loss {losses.val:.4f} ({losses.avg:.4f})\t'
          'MSE_Loss {MSE_losses.val:.4f} ({MSE_losses.avg:.4f})'.format( losses.epoch,
           losses.batch, batch_length, batch_time=times, losses=losses, MSE_losses=MSE_losses))

def train(model, batch, loss_fn, MSE_loss_fn, optimizer):
    model.train() # switch to train mode

    lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))
    values = Variable((batch['value']).type(torch.cuda.FloatTensor))
    targets = batch['output']

    # DONE kolla så classification_targets ser bra ut och är vänt åt rätt håll..
    classification_targets = get_ground_truth(args.resolution,(targets),radius=args.radius)
    classification_targets = Variable(classification_targets).type(torch.cuda.FloatTensor)

    center_of_mass = get_coords_from_predictions(classification_targets,args.resolution)
    #print(classification_targets.shape)
    #print(targets.shape)
    #print(targets[0])
    #print(center_of_mass[0])
    #plt.imshow(classification_targets[0,5,:,:])
    #plt.show()

    output = model(lidars, values) # batch x layer x witdh x height
    #fig, axes = plt.subplots()
    #os = output[0,:,:,:].data.cpu().numpy()
    #axes[].plot(x, y)
    #plt.imshow(os[0])
    #plt.show()
    #plt.imshow(output.data[0][0])
    #plt.show()
    loss = loss_fn(output, classification_targets)

    # TODO titta i matla hur outputen ser ut.
    center_of_masses = get_coords_from_predictions(output,args.resolution)
    # TODO plotta var center of masses är jämfört med targets.
    MSE_loss = MSE_loss_fn(torch.tensor(center_of_masses), targets)

    optimizer.zero_grad() # reset gradients
    loss.backward()
    optimizer.step() # update weights

    return loss.data[0],MSE_loss.data[0] # return loss for this batch

def validate(model, dataloader, loss_fn, MSE_loss_fn, save_output=False):
    model.eval() # switch to eval mode
    losses = ResultMeter()
    MSE_losses = ResultMeter()

    for i, batch in enumerate(dataloader):
        # Read input and output into variables
        lidars = Variable((batch['lidar']).type(torch.cuda.FloatTensor))#,volatile=True)
        values = Variable((batch['value']).type(torch.cuda.FloatTensor))#,volatile=True)
        targets = batch['output']

        classification_targets = get_ground_truth(args.resolution,targets,radius=args.radius)
        classification_targets = Variable(classification_targets).type(torch.cuda.FloatTensor)

        indices = batch['index']

        # Run model and calculate loss
        output = model(lidars, values)
        loss = loss_fn(output, classification_targets)

        center_of_masses = get_coords_from_predictions(output,args.resolution)
        MSE_loss = MSE_loss_fn(torch.tensor(center_of_masses), targets)

        # Save generated predictions to file
        if save_output:
            current_batch_size = numpy.size(output,0)
            # Split the arrays in batch examples.
            center_of_masses = numpy.split(center_of_masses, current_batch_size, 0)
            semantic_segmentation = numpy.split(output.data, current_batch_size, 0)

            path = PATH_SAVE + 'generated_output/'
            generate_output(indices, center_of_masses, semantic_segmentation, batch['foldername'], path)

        # Document results
        losses.update(0, 0, i, loss.data[0])
        MSE_losses.update(0, 0, i, MSE_loss.data[0])

        # Print statistics
        if i % args.print_freq == 0:
            print('Validation: [{batch}/{total}]\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'MSE_Loss {MSE_losses.val:.4f} ({MSE_losses.avg:.4f})'.format( batch = i,
                    total = len(dataloader), losses=losses, MSE_losses=MSE_losses))

    return losses.avg, MSE_losses.avg

def get_coords_from_predictions(output,resolution):


    #center_of_masses = [[prediction_to_coords(resolution, layer.cpu().numpy()) for layer in example] for example in output.data]
    #center_of_masses = numpy.array(center_of_masses)
    #center_of_masses = numpy.reshape(center_of_masses, (len(center_of_masses), -1))
    #batch
    batch_coords =[]
    for example in output.data:
        #example from batch
        x_coords = []
        y_coords = []
        for layer in example:
            x,y = prediction_to_coords(resolution, layer.cpu().numpy())
            x_coords.append(x)
            y_coords.append(y)
        coords = x_coords + y_coords
        batch_coords.append(coords)
    center_of_masses = numpy.array(batch_coords)
    return center_of_masses

def prediction_to_pixel_coords(resolution, prediction):
    # Set all predicted values below 0.5 to 0
    #prediction[numpy.where(prediction < 0.5)] = 0

    # Calculate the center of mass of the remaining predicted values

    y,x = ndimage.measurements.center_of_mass(prediction)
    if numpy.isnan([x,y]).any():
        # There were only zeros in the predictions, so center_of_mass could not
        # handle it. Let's define it as having center of mass equal to (0,0)
        width, height = resolution
        x = width/2
        y = height/2

    return x,y

def prediction_to_coords(resolution, layer, roi=60):
    px, py = prediction_to_pixel_coords(resolution, layer)
    # px och py är pixelkoordinater som inte är avrundade.

    # Convert from pixel coordinate system to vehicle relative system
    width, height = resolution
    # gör om till relativa koordinater (i meter)
    x = (px-(width/2))/(width/roi)
    y = -(py-(height/2))/(height/roi)
    return [x,y]

def make_circle(width, height, x, y, radius, roi):
    # Always 60x60 meters with car in middle. Coordinates are relative so up is positive y and right is positive x.
    #init array with zeros
    arr = numpy.zeros((width,height))
    # beräkna vilken pixel det motsvarar
    px = round(width/2 + x*(width/roi))
    py = round(height/2 - y*(height/roi))
    # draw circle with numpy
    rows,cols = draw.circle(py,px,radius=radius,shape=(width,height)) # py before px as it takes the arguments rows,cols.
    arr[rows,cols] = 1
    # få ut boolean array.
    return arr

def get_ground_truth(size, pos, radius):
    pos = numpy.array(pos)
    #pos = [[x,y] for x,y in pos.transpose(1,0)]
    width, height = size
    roi = 60 #always 60x60 meters with car in middle
    pixel_radius = round(radius*(width/roi))
    batch = []
    for arr in pos:
        coords = numpy.reshape(arr, (2,-1)).transpose(1,0)
        arrays = [make_circle(width,height, x, y, pixel_radius,roi) for x,y in coords]  # layers x width x height
        arrays = numpy.stack(arrays, axis=0)
        batch.append(arrays)
    ground_truth = numpy.stack(batch, axis=0) # batch x steps x width x height
    return torch.tensor(ground_truth)

def generate_output(indices, outputs, semantic_segmentation, foldername, path):

    if not os.path.exists(path):
        os.makedirs(path)

    for i, output in enumerate(outputs):
        subpath_coords = path + str(foldername[i]) + '/coords/'
        if not os.path.exists(subpath_coords):
            os.makedirs(subpath_coords)

        output = numpy.reshape(output, (2,-1)).transpose(1,0) # format the output
        filename = subpath_coords + 'gen_%i.csv' %(int(indices[i]))
        numpy.savetxt(filename, output, comments='', delimiter=',',fmt='%.8f',
                        header='x,y')

    for i, sem_seg in enumerate(semantic_segmentation):
        subpath_semseg = path + str(foldername[i]) + '/semantic_segmentation/'
        if not os.path.exists(subpath_semseg):
            os.makedirs(subpath_semseg)

        sem_seg = sem_seg.data[0].cpu().numpy() # convert from tensor to np array

        for j, ss in enumerate(sem_seg):
            filename = subpath_semseg + 'gen_%i_%i.csv' %(int(indices[i]),j)
            numpy.savetxt(filename, ss, comments='', delimiter=',')

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

def write_test_loss_file(class_loss, mse_loss):
    loss_txt = 'Class loss: %.5f \n MSE loss: %.5f' %(class_loss, mse_loss)
    file = open(PATH_SAVE + "test_loss.txt", "w")
    file.write(loss_txt)
    file.close()

if __name__ == '__main__':
    main()
