import os
import re
import numpy as np

def getData(path,max):
    print("Loading: " + path)
    path_topviews = path + 'input/topviews/max_elevation/'
    path_values = path + 'input/values/'
    path_output = path + 'output/'

    dic = {}
    print('\tINDICES')
    indices = getIndices(path_topviews, max)
    dic['indices'] = indices
    print('\tLIDAR')
    dic['lidar'] = getArray(path_topviews, 600, 600, False, indices, 'me_')
    print('\tVALUES')
    dic['values'] = getArray(path_values, 30, 11, True, indices, 'input_')
    print('\tOUTPUT')
    dic['output'] = getArray(path_output, 30, 2, True, indices, 'output_')
    return dic

def getArray(path, h, w, header, indices, filename):
    # count files
    nr_of_files = len(os.listdir(path))
    max = len(indices)
    res = np.zeros([max,h,w])
    idx = 0
    for i in indices:
        if idx>=max:
            return res
        data = np.genfromtxt(path+filename+'%i.csv' %i, delimiter=',', skip_header=header)
        data = np.nan_to_num(data)
        res[idx] = data
        idx = idx + 1
        if(idx%100==0):
            print('\tindex: %i of max %i, filetotal is %i' %(idx, max, nr_of_files) )
    return res

def getIndices(path, max):
    nr_of_files = len(os.listdir(path))
    res = np.zeros([max])
    idx = 0
    for filename in os.listdir(path):
        if idx>=max:
            return res
        data = int (re.search('\d+', filename).group())
        res[idx] = data
        idx = idx + 1
        if(idx%100==0):
            print('\tindex: %i of max %i, filetotal is %i' %(idx, max, nr_of_files) )
    return res
