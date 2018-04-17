# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def main():
    DATASET_PATH = '/media/annaochjacob/crucial/dataset/fruit_salad/peach_split/'
    INPUT_PATH = DATASET_PATH + 'test/input/values/'
    OUTPUT_PATH = DATASET_PATH + 'test/output/'

    n_files = 0
    total_error = 0
    for input_filename in os.listdir(INPUT_PATH):
        filename = INPUT_PATH + input_filename
        input_v = np.loadtxt(filename, delimiter=',', skiprows=1)
        forward_speed = input_v[0,3]

        output_filename = (re.sub('input_','output_',input_filename))
        filename = OUTPUT_PATH + output_filename
        output_v = np.loadtxt(filename, delimiter=',', skiprows=1)

        baseline = get_baseline(forward_speed, 30, 0.1)
        total_error += mse(output_v,baseline)
        n_files += 1

    average_error = total_error / n_files
    print('Number of files: %i' % n_files)
    print('Average error: %f' % average_error)


def plot(output, baseline):
    plot = plt.scatter(output[:,0], output[:,1], marker='.', c='g')
    plot = plt.scatter(baseline[:,0], baseline[:,1], marker='.', c='y')
    fig = plt.gcf()
    axes = fig.gca()
    axes.set_xlim([-30, 30])
    axes.set_ylim([-30, 30])
    axes.set_aspect('equal')
    plt.show()

def get_baseline(forward_speed, n_points, time_delta):
    y = np.arange(n_points) + 1
    y = y*(forward_speed*time_delta)
    x = np.zeros(n_points)
    positions = np.vstack((x,y)).transpose()
    return positions

def mse(a, b):
    '''Calculates the element-wise mean squared error of a and b'''
    return np.sum(np.square(a-b)) / a.size

if __name__ == '__main__':
    main()
