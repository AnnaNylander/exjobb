import os
import re
import numpy as np

def getData(path, past_frames, max=-1,):
    print("Loading: " + path)
    path_topviews = path + 'input/topviews/max_elevation/'
    path_values = path + 'input/values/'
    path_output = path + 'output/'
    if max > len(os.listdir(path_topviews)) or max ==-1:
        max = len(os.listdir(path_topviews))

    dic = {}
    indices = getIndices(path_topviews, max)
    dic['indices'] = indices
    dic['lidar'] = getLidarFrames(path, indices, past_frames)
    dic['values'] = [path+'input/values/input_%i.csv' %i for i in indices]
    dic['output'] = [path+'output/output_%i.csv' %i for i in indices]

    return dic

def getLidarFrames(path, indices, past_frames):
    res = []
    subdir = 'input/topviews/max_elevation/'
    categories = 'straight|left|right|right_intention|left_intention|traffic_light|other'
    base_path = re.search('.*(?=(' + categories + ')\/\Z)',path).group(0)
    folders = ['train/', 'validate/', 'test/']
    for i in indices:
        frames = []
        frames.append(path+'input/topviews/max_elevation/me_%i.csv' %i) #main lidar picture. The current one.
        # past lidar pictures
        for j in past_frames:
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
        if idx >= max:
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
        if idx >= max:
            return sorted(res)
        data = int (re.search('\d+', filename).group())
        res[idx] = data
        idx = idx + 1
        #if(idx%100==0):
        #    print('\tindex: %i of max %i, filetotal is %i' %(idx, max, nr_of_files) )
    return res

def get_sampled_data(data_path, past_frames, step_dict, max_limit=None):
    '''Creates a dictionary of sampled data, where step_dict is a dictionary
    from category to step size, e.g. {'left':2} to keep every other frame in
    category 'left'. Parameter max_limit tells how many indices to keep in each
    category. The first max_limit frames will be used. If there are less than
    max_limit,only the frames available are used, i.e. no resampling.

    Example of step_dict:
    1 means keep every, 2 means keep every other, and so on...
    0 means do not add any from this category

    step_dict = {'straight' : 1,
                 'left' : 1,
                 'right' : 1,
                 'right_intention' : 1,
                 'left_intention' : 1,
                 'traffic_light' : 1,
                 'other' : 0
                 }
    '''

    sampled_data = {}

    # for each category in banana
    for category in step_dict.keys():

        # Data has the following structure:
        # key in indices, values, lidars, output
        cat_data = getData(data_path + category + '/', [])

        # Get the number of frames in this category
        n_frames = len(cat_data['indices'])

        # Get the stepsize for this category
        step = step_dict[category]

        # Check for empy categories
        is_empty_category = len(cat_data['indices']) == 0

        # If there is a non-zero step size, pick out the frames
        if step != 0 and not is_empty_category:
            asorted_indices = cat_data['indices']

            # Pick out the indices to keep from inputs, lidar, values and output
            for key, value in cat_data.items():

                # Sort indices, lidar, values and output based on indices
                value = [x for _,x in sorted(zip(asorted_indices, value))]

                # Keep every 'step'th element
                data_to_keep = value[0::step]

                # Cap length of data list if longer than max_limit
                if max_limit is not None and len(data_to_keep) > max_limit:
                    data_to_keep = data_to_keep[0:max_limit]

                # Append current data to all previously sampled data
                if key in sampled_data:
                    data_to_keep = np.concatenate((sampled_data[key], data_to_keep), axis=0)

                sampled_data[key] = data_to_keep

    return sampled_data
