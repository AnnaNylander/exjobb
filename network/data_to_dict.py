import os
import re
import numpy as np

def get_category_data(path, past_frames, max_limit=-1):
    path_topviews = path + 'input/topviews/max_elevation/'
    path_values = path + 'input/values/'
    path_output = path + 'output/'
    if max_limit > len(os.listdir(path_topviews)) or max_limit ==-1:
        max_limit = len(os.listdir(path_topviews))

    dic = {}
    indices = getIndices(path_topviews, max_limit)
    dic['indices'] = indices
    dic['lidar'] = getLidarFrames(path, indices, past_frames)
    dic['values'] = [path+'input/values/input_%i.csv' %i for i in indices]
    dic['output'] = [path+'output/output_%i.csv' %i for i in indices]

    return dic

def get_episode_data(episode_path, past_frames, max_limit=-1):
    categories = os.listdir(episode_path)
    episode_data = {}

    for category in categories:
        path = episode_path + category + '/'
        cat_data = get_category_data(path, past_frames, max_limit)
        n_frames = len(cat_data['indices'])
        cat_data['category'] = [category]*n_frames

        # Append frames if there are any
        if n_frames > 0:
            for key, value in cat_data.items():
                if key in episode_data:
                    episode_data[key] = np.concatenate((episode_data[key], value), axis=0)
                else:
                    episode_data[key] = value

    # Sort all frames by index
    asorted_indices = episode_data['indices']
    for key in list(episode_data.keys()):
        values = episode_data[key]
        sorted_values = [x for _,x in sorted(zip(asorted_indices, values))]
        episode_data[key] = np.asarray(sorted_values)

    return episode_data


def get_data(data_path, past_frames, max_limit=-1):
    datasets = os.listdir(data_path)
    data = {}
    for dataset in datasets:
        print('Loading', data_path + dataset)
        path = data_path + dataset + '/'
        episode_data = get_episode_data(path, past_frames, max_limit)

        for key, value in episode_data.items():
            if key in data:
                data[key] = np.concatenate((data[key], value), axis=0)
            else:
                data[key] = value

    return data

def getLidarFrames(path, indices, past_frames):
    res = []
    subdir = 'input/topviews/max_elevation/'
    #['straight/','left/','right/','right_intention/','left_intention/','traffic_light/','other/']
    folders = [string + '/' for string in os.listdir(path + '..')]
    base_path = re.search('.*(?=(' + '|'.join(os.listdir(path+'..')) + ')\/\Z)',path).group(0)

    for i in indices:
        frames = []
        frames.append(path+'input/topviews/max_elevation/me_%i.csv' %i) # main lidar picture. The current one.
        for j in past_frames:
            idx = i - j
            frame = ''
            while frame == '' and idx <= i:
                for folder in folders:
                    filename = base_path + folder + 'input/topviews/max_elevation/me_%i.csv' %idx
                    if os.path.isfile(filename):
                        frame = filename
                        break
                idx += 1
            if frame=='':
                print("WARNING: Not found " + frame)
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
        if idx >= max:
            return res
        data = np.genfromtxt(path+filename+'%i.csv' %i, delimiter=',', skip_header=header)
        data = np.nan_to_num(data)
        res[idx] = data
        idx = idx + 1

    return res

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
