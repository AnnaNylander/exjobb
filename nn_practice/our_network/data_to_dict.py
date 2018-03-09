import os
import numpy as np

def getData(path):
    path_topviews = path + 'input/topviews/max_elevation/'
    path_values = path + 'input/values/'
    path_output = path + 'output/'

    dic = {}
    dic['lidar'] = getArray(path_topviews)
    dic['values'] = getArray(path_values)
    dic['output'] = getArray(path_output)
    return dic

def getArray(path):
    # count files
    nr_of_files = len(os.listdir(path))
    res = None
    for filename in os.listdir(path):
        data = [np.genfromtxt(path+filename, delimiter=',', names=True)]
        if res is None:
            res = data
        else:
            res = np.append(res,data, axis=0)
    return res
