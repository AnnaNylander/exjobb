import numpy as np
import matplotlib.pyplot as plt
import argparse

SIDE = 40
DATA_PATH = '/Users/Jacob/Documents/Datasets/exjobb/recorded_data_2018-03-02/data_set'
FACECOLOR = 'k'
SUBPLOT_ROWS = 2
SUBPLOT_COLS = 2

parser = argparse.ArgumentParser(description='Plot positions relative to car')
parser.add_argument('--step', metavar='integer', type=int,
                    dest='timeStep', default='0',
                    help='Time step (frame index) to plot as current fram.')

args = parser.parse_args()


def main():
    output_path = DATA_PATH + '/output/output_%i.csv' % args.timeStep
    input_path = DATA_PATH + '/input/input_%i.csv' % args.timeStep

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
    plot(future_loc_x, future_loc_y, 'b', SUBPLOT_ROWS, SUBPLOT_COLS, \
        1,FACECOLOR,'Past and future path',SIDE)
    plot(past_loc_x, past_loc_y, 'r', SUBPLOT_ROWS, SUBPLOT_COLS, \
        1,FACECOLOR,'Past and future path',SIDE)
    plot(future_loc_x, future_loc_y, past_fwd_acc, SUBPLOT_ROWS, SUBPLOT_COLS, \
        2,FACECOLOR,'Forward acceleration',SIDE)
    plot(future_loc_x, future_loc_y, past_fwd_speed, SUBPLOT_ROWS, SUBPLOT_COLS, \
        3,FACECOLOR,'Forward speed',SIDE)
    plot(future_loc_x, future_loc_y, past_steer, SUBPLOT_ROWS, SUBPLOT_COLS, \
        4,FACECOLOR,'Steer',SIDE)

    plt.show()

def plot(x, y ,c , subplot_rows, subplot_cols, subplot_index, facecolor, title, side):
    plt.subplot(subplot_rows, subplot_cols, subplot_index, facecolor=facecolor)
    future_plot = plt.scatter(x, y, marker='.', c=c)
    fig = plt.gcf()
    #plt.legend((future_plot, past_plot),('Future','Past'))
    axes = fig.gca()
    axes.set_xlim([-side/2, side/2])
    axes.set_ylim([-side/2, side/2])
    axes.set_aspect('equal')
    axes.set_title(title)

if __name__ == "__main__":
    main()
