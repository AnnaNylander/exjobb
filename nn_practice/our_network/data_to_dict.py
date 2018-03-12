import os
import numpy as np

def getData(path,max):
    print("Loading: " + path)
    path_topviews = path + 'input/topviews/max_elevation/'
    path_values = path + 'input/values/'
    path_output = path + 'output/'

    dic = {}
    print('\tLIDAR')
    dic['lidar'] = getArray(path_topviews, 600, 600, False, max)
    print('\tVALUES')
    dic['values'] = getArray(path_values, 30, 11, True, max)
    print('\tOUTPUT')
    dic['output'] = getArray(path_output, 30, 2, True, max)
    return dic

def getArray(path, h, w, header, max):
    # count files
    nr_of_files = len(os.listdir(path))
    res = np.zeros([max,h,w])
    idx = 0
    for filename in os.listdir(path):
        if idx>=max:
            return res
        data = np.genfromtxt(path+filename, delimiter=',', skip_header=header)
        data = np.nan_to_num(data)
        res[idx] = data
        idx = idx + 1
        if(idx%100==0):
            print('\tindex: %i of max %i, filetotal is %i' %(idx, max, nr_of_files) )
    return res
