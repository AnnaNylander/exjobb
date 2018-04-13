import os
import re
import numpy as np

def getData(path, past_frames, frame_stride, max=-1,):
    print("Loading: " + path)
    path_topviews = path + 'input/topviews/max_elevation/'
    path_values = path + 'input/values/'
    path_output = path + 'output/'
    if max > len(os.listdir(path_topviews)) or max ==-1:
        max = len(os.listdir(path_topviews))

    dic = {}
    indices = getIndices(path_topviews, max)
    dic['indices'] = indices #['%s_%i' %(path,i) for i in indices]

    #dic['lidar'] = [path+'input/topviews/max_elevation/me_%i.csv' %i for i in indices]  #path for lidar picture

    dic['lidar'] = getLidarFrames(path, indices, past_frames, frame_stride)

    dic['values'] = [path+'input/values/input_%i.csv' %i for i in indices]
    #dic['values'] = getArray(path_values, 30, 11, True, indices, 'input_')
    dic['output'] = [path+'output/output_%i.csv' %i for i in indices]
    #dic['output'] = getArray(path_output, 30, 2, True, indices, 'output_')
    return dic

def getLidarFrames(path, indices, past_frames, frame_stride):
    res = []
    subdir = 'input/topviews/max_elevation/'
    base_path = re.search('.*(?=(train|validate|test)\/\Z)',path).group(0)
    folders = ['train/', 'validate/', 'test/']
    for i in indices:
        frames = []
        frames.append(path+'input/topviews/max_elevation/me_%i.csv' %i) #main lidar picture. The current one.
        # past lidar pictures
        for j in range(1,past_frames*frame_stride+1, frame_stride):
            idx = i - j
            frame = ''
            while frame == '':
                for folder in folders:
                    if os.path.isfile(base_path + folder + 'input/topviews/max_elevation/me_%i.csv' %idx):
                        frame = base_path + folder + 'input/topviews/max_elevation/me_%i.csv' %idx
                idx += 1
            frames.append(frame)
        res.append(frames)
    return res



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
