import os
import re
import numpy as np

def get_data(data_path, past_frames, max_limit=-1):
    datasets = os.listdir(data_path)
    data = {}
    for dataset in datasets:
        print('Loading', data_path + dataset)
        path = data_path + dataset + '/'

        category_data = get_category_data(path, past_frames, max_limit)

        for key, value in category_data.items():
            if key in data:
                data[key] = np.concatenate((data[key], value), axis=0)
            else:
                data[key] = value

    return data

def get_category_data(path, past_frames, max_limit=-1):
    path_lidar = path + 'max_elevation/'
    path_values = path

    if max_limit > len(os.listdir(path_lidar)) or max_limit ==-1:
        max_limit = len(os.listdir(path_lidar))

    dic = {}
    indices = getIndices(path_lidar, max_limit)
    dic['indices'] = indices
    dic['lidar'] = getLidarFrames(path_lidar, indices, past_frames)
    dic['values'] = getValues(path, indices, past_frames)

    return dic

def getLidarFrames(path, indices, past_frames):
    res = []

    for i in indices:
        frames = []
        frames.append(path + 'me_%i.csv' %i) # main lidar picture. The current one.
        for j in past_frames:
            idx = i - j
            frame = ''
            while frame == '' and idx <= i:
                if os.path.isfile(path + 'me_%i.csv' %idx):
                    frame = filename
                    break
                idx += 1
            if frame=='':
                print("WARNING: Not found " + frame)
            frames.append(frame)
        res.append(frames)
    return res

def getValues(path, indices, past_frames):
    # TODO return one row for each frame.
    dic = {}
    return dic

def getIndices(path, max):
    nr_of_files = len(os.listdir(path))
    res = np.zeros([max,1])
    idx = 0
    for filename in os.listdir(path):
        if idx >= max:
            return sorted(res)
        data = int (re.search('\d+', filename).group())
        res[idx] = data
        idx = idx + 1

    return res
