import numpy as np
import matplotlib.pyplot as plt
import argparse

SIDE = 40
DATA_PATH = '/Users/Jacob/Documents/Datasets/exjobb/recorded_data_2018-03-02/data_set'

parser = argparse.ArgumentParser(description='Plot positions relative to car')
parser.add_argument('--step', metavar='integer', type=int,
                    dest='timeStep', default='0',
                    help='Time step (frame index) to plot as current fram.')

args = parser.parse_args()

output_path = DATA_PATH + '/output/output_%i.csv' % args.timeStep
input_path = DATA_PATH + '/input/input_%i.csv' % args.timeStep

loc_future = np.genfromtxt(output_path, delimiter=',', skip_header=True)
loc_past = np.genfromtxt(input_path, delimiter=',', skip_header=True)

future_plot = plt.scatter(loc_future[:,0],loc_future[:,1],marker='.', c='b')
past_plot = plt.scatter(loc_past[:,0],loc_past[:,1],marker='.', c='r')

fig = plt.gcf()
plt.legend((future_plot, past_plot),('Future','Past'))
axes = fig.gca()
axes.set_xlim([-SIDE/2, SIDE/2])
axes.set_ylim([-SIDE/2, SIDE/2])
axes.set_aspect('equal')

plt.show()
