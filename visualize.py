import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from lidar_to_topview.main import lidar_to_topview, get_max_elevation

ROI = 60
CELLS = 600
SIDE = 40
DATA_PATH = '/Users/Jacob/Documents/Datasets/exjobb/recorded_data_2018-03-02/data_set'
PATH_POINT_CLOUD = '/Users/Jacob/Documents/Datasets/exjobb/recorded_data_2018-03-02/point_cloud'
SUBPLOT_ROWS = 2
SUBPLOT_COLS = 2

parser = argparse.ArgumentParser(description='Plot positions relative to car')
parser.add_argument('--step', metavar='integer', type=int,
                    dest='timeStep', default=0,
                    help='Time step (frame index) to plot as current fram.')

args = parser.parse_args()

def main():
    output_path = DATA_PATH + '/output/output_%i.csv' % args.timeStep
    input_path = DATA_PATH + '/input/values/input_%i.csv' % args.timeStep

    # Read point cloud and generate top view image
    filename = PATH_POINT_CLOUD + '/pc_%i.csv' % args.timeStep
    point_cloud = np.genfromtxt(filename, delimiter=' ')
    topview_img = lidar_to_topview(args.timeStep, point_cloud, ROI, CELLS)
    topview_img = topview_img.squeeze()
    #imgplot = plt.imshow(topview_img, cmap='gray')
    #plt.show()

    values_future = np.genfromtxt(output_path, delimiter=',', skip_header=True)
    values_past = np.genfromtxt(input_path, delimiter=',', skip_header=True)

    future_loc_x = values_future[:,0]
    future_loc_y = values_future[:,1]
    past_loc_x = values_past[:,0]
    past_loc_y = values_past[:,1]
    past_fwd_acc = values_past[:,2]
    past_fwd_speed = values_past[:,3]
    past_steer = values_past[:,4]

    # Past and future path
    plot(future_loc_x, future_loc_y, 'b', topview_img, SUBPLOT_ROWS, \
    SUBPLOT_COLS, 1,'Past and future path',SIDE)
    plot(past_loc_x, past_loc_y, 'r', topview_img, SUBPLOT_ROWS, \
        SUBPLOT_COLS, 1,'Past and future path',SIDE)
    plot(past_loc_x, past_loc_y, past_fwd_acc, topview_img, SUBPLOT_ROWS, \
        SUBPLOT_COLS, 2,'Forward acceleration',SIDE)
    plot(past_loc_x, past_loc_y, past_fwd_speed, topview_img, SUBPLOT_ROWS, \
        SUBPLOT_COLS, 3,'Forward speed',SIDE)
    plot(past_loc_x, past_loc_y, past_steer, topview_img, SUBPLOT_ROWS, \
        SUBPLOT_COLS, 4,'Steer',SIDE)

    plt.show()

def plot(x, y ,c , image ,subplot_rows, subplot_cols, subplot_index, title, side):
    plt.subplot(subplot_rows, subplot_cols, subplot_index)
    plt.imshow(image, cmap='gray', extent=[-side/2, side/2, -side/2, side/2], \
        interpolation='bilinear')
    cmap = matplotlib.cm.Greys
    future_plot = plt.scatter(x, y, marker='.', c=c, cmap=cmap)
    fig = plt.gcf()
    #plt.legend((future_plot, past_plot),('Future','Past'))
    axes = fig.gca()
    axes.set_xlim([-side/2, side/2])
    axes.set_ylim([-side/2, side/2])
    axes.set_aspect('equal')
    axes.set_title(title)

if __name__ == "__main__":
    main()
