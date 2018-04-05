import numpy as np
import matplotlib
import os
import re
import matplotlib.pyplot as plt
import argparse
from preprocessing.util import lidar_to_topview, get_max_elevation,trim_to_roi

parser = argparse.ArgumentParser(description='Save a topview/plot with past, predicted, and future ground-truth path.')
parser.add_argument('--step', metavar='N', type=int,
                    dest='timeStep', default=None,
                    help='Time step (frame index) to plot. (default all)')
parser.add_argument('--prediction', dest='prediction', action='store_true',
                    help='Whether to plot predictions or not')
parser.add_argument('--everything', dest='everything', action='store_true',
                    help='Whether to plot all graphs or not.')

parser.add_argument('-s','--save-path', metavar='path',
                    dest='save_path',
                    help='Where to save the images. THE FULL PATH.')

parser.add_argument('-d','--dataset', metavar='path',
                    dest='dataset',
                    help='Foldername of dataset. Eg. Banan_split/test/ (with trailing /)')
parser.add_argument('-r','--recorded_data', metavar='path',
                    dest='recorded',
                    help='Foldername of recorded data. Eg. recorded_2018_03_14/ (with trailing /)')
parser.add_argument('-m','--models', metavar='path',
                    dest='saved_models',
                    help='Foldername of model. Eg. LucaNet/ (with trailing /)')
args = parser.parse_args()

ROI = 60
CELLS = 600
SIDE = 60
PATH_BASE = '/media/annaochjacob/crucial/'
PATH_DATA = PATH_BASE +'dataset/'+ args.dataset
PATH_POINT_CLOUD = PATH_BASE +'recorded_data/carla/'+ args.recorded + 'point_cloud/'
PATH_PREDICTION = PATH_BASE +'models/'+ args.saved_models + 'generated_output/'
SAVE_PATH = args.save_path
SUBPLOT_ROWS = 1
SUBPLOT_COLS = 1

def main():
    # if only one timeStep
    if args.timeStep is not None:
        visualize(args.timeStep, SAVE_PATH, args.prediction, args.everything)

    # else save all in folder.
    else:
        indices = getIndices(PATH_DATA)
        indices = sorted(indices)
        for i in indices:
            visualize(i, SAVE_PATH, args.prediction, args.everything)
            print(i)

def getIndices(path):
    path = path + 'input/values/'
    nr_of_files = len(os.listdir(path))
    res = np.zeros([nr_of_files])
    idx = 0
    for filename in os.listdir(path):
        data = int (re.search('\d+', filename).group())
        res[idx] = data
        idx = idx + 1
    return res


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
    axes.set_title(title)

def visualize(step, path, prediction, everything):
    plt.gcf().clear()
    SUBPLOT_ROWS = 1
    SUBPLOT_COLS = 1
    if everything:
        SUBPLOT_ROWS = 2
        SUBPLOT_COLS = 2

    # Read point cloud and generate top view image
    filename = PATH_POINT_CLOUD + 'pc_%i.csv' % step
    point_cloud = np.genfromtxt(filename, delimiter=',', skip_header=True)
    point_cloud = trim_to_roi(point_cloud, ROI)
    topview_img = lidar_to_topview(point_cloud, ROI, CELLS)
    topview_img = topview_img.squeeze()

    #past
    path_input = PATH_DATA + 'input/values/input_%i.csv' % step
    values_past = np.genfromtxt(path_input, delimiter=',', skip_header=True)
    past_loc_x = values_past[:,0]
    past_loc_y = values_past[:,1]
    plot_past = plot(past_loc_x, past_loc_y, 'r', topview_img, SUBPLOT_ROWS, SUBPLOT_COLS, 1,'Past and future path',SIDE)

    #future
    path_output = PATH_DATA + 'output/output_%i.csv' % step
    values_future = np.genfromtxt(path_output, delimiter=',', skip_header=True)
    future_loc_x = values_future[:,0]
    future_loc_y = values_future[:,1]
    plot_future = plot(future_loc_x, future_loc_y, 'g', topview_img, SUBPLOT_ROWS, SUBPLOT_COLS, 1,'Past and future path',SIDE)

    #prediction
    if prediction:
        path_pred = PATH_PREDICTION + 'gen_%i.csv' % step
        values_pred = np.genfromtxt(path_pred, delimiter=',', skip_header=True)
        pred_loc_x = values_pred[:,0]
        pred_loc_y = values_pred[:,1]
        plot_pred = plot(pred_loc_x, pred_loc_y, 'y', topview_img, SUBPLOT_ROWS, SUBPLOT_COLS, 1,'Predicted',SIDE)

    # if several subplots
    if everything:
        past_fwd_acc = values_past[:,2]
        past_fwd_speed = values_past[:,3]
        past_steer = values_past[:,4]
        plot(past_loc_x, past_loc_y, past_fwd_acc, topview_img, SUBPLOT_ROWS, SUBPLOT_COLS, 2,'Forward acceleration',SIDE)
        plot(past_loc_x, past_loc_y, past_fwd_speed, topview_img, SUBPLOT_ROWS,SUBPLOT_COLS, 3,'Forward speed',SIDE)
        plot(past_loc_x, past_loc_y, past_steer, topview_img, SUBPLOT_ROWS, SUBPLOT_COLS, 4,'Steer',SIDE)

    plt.savefig(path + 'img_%i.png' %step)

if __name__ == "__main__":
    main()
