import os
import re
import numpy as np

def getData(path,max=-1):
    print("Loading: " + path)
    path_topviews = path + 'input/topviews/max_elevation/'
    path_values = path + 'input/values/'
    path_output = path + 'output/'
    if max > len(os.listdir(path_topviews)) or max ==-1:
        max = len(os.listdir(path_topviews))

    dic = {}
    #print('\tINDICES')
    indices = getIndices(path_topviews, max)
    dic['indices'] = indices #['%s_%i' %(path,i) for i in indices]
    #print('\tLIDAR')
    dic['lidar'] = [path+'input/topviews/max_elevation/me_%i.csv' %i for i in indices]  #path for lidar picture
    #print('\tVALUES')
    dic['values'] = [path+'input/values/input_%i.csv' %i for i in indices]
    #dic['values'] = getArray(path_values, 30, 11, True, indices, 'input_')
    #print('\tOUTPUT')
    dic['output'] = [path+'output/output_%i.csv' %i for i in indices]
    #dic['output'] = getArray(path_output, 30, 2, True, indices, 'output_')
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
        #if(idx%100==0):
        #    print('\tindex: %i of max %i, filetotal is %i' %(idx, max, nr_of_files) )
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
        #if(idx%100==0):
        #    print('\tindex: %i of max %i, filetotal is %i' %(idx, max, nr_of_files) )
    return res
