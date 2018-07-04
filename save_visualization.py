import numpy as np
import matplotlib
import os
import re
import matplotlib.pyplot as plt
import argparse
from preprocessing.util import lidar_to_topview, get_max_elevation, world_to_relative

parser = argparse.ArgumentParser(description='Save a topview/plot with past, predicted, and future ground-truth path.')
parser.add_argument('--step', metavar='N', type=int,
                    dest='time_step', default=None,
                    help='Time step (frame index) to plot. (default all)')
parser.add_argument('--prediction', dest='prediction', action='store_true',
                    help='Whether to plot predictions or not')
parser.add_argument('--everything', dest='everything', action='store_true',
                    help='Whether to plot all graphs or not.')
parser.add_argument('-d','--dataset', metavar='path',
                    dest='dataset',
                    help='Foldername of dataset, e.g. flexible/test/ (with trailing /)')
parser.add_argument('-r','--recorded_data', metavar='path',
                    dest='recorded',
                    help='Foldername of recorded data in carla, e.g. test/ (with trailing /)')
parser.add_argument('-m','--models', metavar='path',
                    dest='saved_models',
                    help='Foldername of model, e.g. LucaNet/ (with trailing /)')
parser.add_argument('-ipf', '--input_past_frames', default='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30', type=str,
                    metavar='\'1 2 3\'', help = 'Which past frames we want to plot. \
                    List which frames you want manually. Ex: \'1 3 5 7 10 13 16\'')
parser.add_argument('-off', '--output_future_frames', default='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30', type=str,
                    metavar='\'1 2 3\'', help = 'Which future frames we want to plot. \
                    List which frames you want manually. Ex: \'1 3 5 7 10 13 16\''),
parser.add_argument('-s','--subset', metavar='name',
                    dest='subset',
                    help='Foldername of subset (without trailing /) to use with individual steps, e.g. birb')
args = parser.parse_args()

ROI = 60
CELLS = 600
SIDE = 60
PATH_BASE = '/media/annaochjacob/crucial/'
PATH_DATA = PATH_BASE +'dataset/'+ args.dataset
PATH_RECORDED_DATA = PATH_BASE +'recorded_data/carla/'+ args.recorded
PATH_MODELS = PATH_BASE +'models/'+ args.saved_models
SUBPLOT_ROWS = 1
SUBPLOT_COLS = 1

def main():

    # Make numpy arrays of past and future indices
    past_idx = -np.array([int(i) for i in args.input_past_frames.split(' ')])
    future_idx = np.array([int(i) for i in args.output_future_frames.split(' ')])

    # if only one time step
    if args.time_step is not None:
        visualize(args.time_step, past_idx, future_idx, args.prediction,
                    args.everything, folder=args.subset)
        return

    # else save all
    for folder in os.listdir(PATH_DATA):
        n_frames = get_number_of_steps(PATH_DATA + folder + '/values.csv')

        # Save visualizations of all time steps in the current folder
        for i in range(0,n_frames+1):
            visualize(i, past_idx, future_idx, args.prediction, args.everything,
                        folder=folder+'/')

def get_number_of_steps(values_path):
    ''' Returns the number of value rows in a given values file. '''
    values = np.loadtxt(values_path, delimiter=',', skiprows=1)
    return len(values)

def plot(x, y ,c , image ,subplot_rows, subplot_cols, subplot_index, title, side):
    plt.subplot(subplot_rows, subplot_cols, subplot_index)
    plt.imshow(image, cmap='gray', extent=[-side/2, side/2, -side/2, side/2], \
        interpolation='bilinear')
    cmap = matplotlib.cm.Greys
    plot = plt.scatter(x, y, marker='.', c=c, cmap=cmap)
    # TODO add legend
    fig = plt.gcf()
    axes = fig.gca()
    axes.set_xlim([-side/2, side/2])
    axes.set_ylim([-side/2, side/2])
    axes.set_aspect('equal')
    #axes.set_title(title)
    plt.axis('off')

def get_values(values_path, current_step, other_steps):
    ''' Returns the values at the given time steps.
    The current step is not automatically included in the output.'''
    values = np.loadtxt(values_path, delimiter=',', skiprows=1)
    return values[current_step + np.array(other_steps), :]

def get_relative_coordinates(values_path, current_step, other_steps):
    ''' Returns the x and y coordinates at the given time steps.
    The current step is not automatically included in the output.'''
    values = np.genfromtxt(values_path, delimiter=',', names=True)
    x = values['location_x'][current_step]
    y = values['location_y'][current_step]
    yaw = values['yaw'][current_step]

    # Pick out the coordinates
    x_coord = values['location_x'][current_step + np.array(other_steps)]
    y_coord = values['location_y'][current_step + np.array(other_steps)]
    w_coord = np.array([x_coord, y_coord])

    # Translate into relative coordinates
    r_coord= world_to_relative(x, y, yaw, w_coord)
    return r_coord

def visualize(current_idx, past_idx, future_idx, prediction, everything, folder=''):

    # Make visualization path if it doesn't exist
    path_save = PATH_MODELS + 'visualization/' + folder + '/'
    if not os.path.exists(path_save): os.makedirs(path_save)

    # if we want prediction but the file doesn't exist, skip and return.
    if prediction:
        path_pred = PATH_MODELS + 'generated_output/' + folder +  'gen_%i.csv' % current_idx
        if not os.path.isfile(path_pred):
            return

    plt.gcf().clear()
    SUBPLOT_ROWS = 1
    SUBPLOT_COLS = 1
    if everything:
        SUBPLOT_ROWS = 2
        SUBPLOT_COLS = 2

    # Read point cloud and generate top view image
    filename = PATH_RECORDED_DATA + folder + 'point_cloud/' + 'pc_%i.csv' % current_idx
    point_cloud = np.genfromtxt(filename, delimiter=',', skip_header=True)
    topview_img = lidar_to_topview(point_cloud, ROI, CELLS)
    topview_img = topview_img.squeeze()

    # Make topview white instead of black
    topview_img = 255 - topview_img

    #future_idx = np.array(list(range(0,31)))
    #past_idx = -future_idx
    path_values = PATH_DATA + folder + '/values.csv'

    #past
    past_coord = get_relative_coordinates(path_values, current_idx, past_idx)
    plot_past = plot(past_coord[0], past_coord[1], 'r', topview_img,
                    SUBPLOT_ROWS, SUBPLOT_COLS, 1,'Past and future path',SIDE)

    #future
    future_coord = get_relative_coordinates(path_values, current_idx, future_idx)
    plot_future = plot(future_coord[0], future_coord[1], 'g', topview_img,
                    SUBPLOT_ROWS, SUBPLOT_COLS, 1,'Past and future path',SIDE)

    #prediction
    if prediction:
        path_pred = PATH_MODELS + 'generated_output/' + folder + 'gen_%i.csv' % current_idx
        values_pred = np.genfromtxt(path_pred, delimiter=',', skip_header=True)
        pred_loc_x = values_pred[:,0]
        pred_loc_y = values_pred[:,1]
        plot_pred = plot(pred_loc_x, pred_loc_y, 'y', topview_img,
                        SUBPLOT_ROWS, SUBPLOT_COLS, 1,'Predicted',SIDE)

    # if several subplots
    if everything:
        past_values = get_values(path_values, current_idx, past_idx)
        acc = past_values[:,3]
        speed = past_values[:,4]
        steer = past_values[:,5]

        plot(past_coord[0], past_coord[1], acc, topview_img, SUBPLOT_ROWS,
                SUBPLOT_COLS, 2,'Total acceleration',SIDE)
        plot(past_coord[0], past_coord[1], speed, topview_img, SUBPLOT_ROWS,
                SUBPLOT_COLS, 3,'Forward speed',SIDE)
        plot(past_coord[0], past_coord[1], steer, topview_img, SUBPLOT_ROWS,
                SUBPLOT_COLS, 4,'Steer',SIDE)

    filepath = path_save + '%s_%i.png' %(re.sub('\/','',folder), current_idx)
    plt.savefig(filepath,bbox_inches='tight',dpi=100)
    print('Saved:', filepath)

if __name__ == "__main__":
    main()
