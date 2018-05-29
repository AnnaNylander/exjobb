import os
import re
import numpy as np

def get_data(data_path, past_frames):
    folders = sorted(os.listdir(data_path))
    data = {}
    for folder in folders:
        print('Loading', data_path + folder)
        path = data_path + folder + '/'

        folder_data = get_folder_data(path, past_frames)
        data[folder] = folder_data

    return data

def get_folder_data(path, past_frames):
    # TODO check lidar images exists.
    path_lidar = path + 'max_elevation/'
    max_limit = len(os.listdir(path_lidar))

    dic = {}
    dic['indices'] = [i for i in range(0,max_limit)]
    dic['values'] = getValues(path)

    return dic

def getValues(path):
    values = np.loadtxt(path + 'values.csv', delimiter=',', skiprows=1) # skip header
    return values
