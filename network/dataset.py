import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import util

class OurDataset(Dataset):

    def __init__(self, data_path, data, no_intention, only_lidar, past_lidars, past_idxs, future_idxs, transform=None):
        """ past_idxs and future_idxs must work like manual_past_frames """

        self.data = data
        global_indices = []
        for folder in data:
            length = len(data[folder]['indices'])
            min_frame = max(past_idxs)
            max_frame = length-1 - max(future_idxs)
            global_indices = global_indices + [(i,folder) for i in range(min_frame,max_frame+1)]

        categories = np.zeros(len(global_indices))
        for i, (idx, folder) in enumerate(global_indices):
            values = data[folder]['values'][idx]
            categories[i] = int(values[12]) # get categories for all rows

        self.categories = categories
        self.data_path = data_path
        self.global_indices = global_indices
        self.no_intention = no_intention
        self.only_lidar = only_lidar
        self.past_idxs = past_idxs
        self.past_lidars = past_lidars
        self.future_idxs = future_idxs
        self.transform = transform

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, global_index):
        (idx, folder) = self.global_indices[global_index]

        # Fetch lidar data.
        lidar = self.get_lidar(idx, folder, self.past_lidars)

        # Fetch values data (input)
        values = self.get_values(idx, folder, self.past_idxs)

        #Fetch output data (output)
        output = self.get_output(idx, folder, self.future_idxs)

        return {'index': idx,
                'lidar': lidar,
                'value': values,
                'output': output,
                'foldername': folder}

    def get_lidar(self, idx, folder, past_lidars):
        path = self.data_path + folder + '/max_elevation/'

        lidars = []
        for i in [0] + past_lidars: # 0 is current frame
            index = idx-i
            lidar = np.genfromtxt(path + 'me_%i.csv' %index, delimiter=',')
            lidars.append(lidar)

        lidars = np.stack(lidars, axis=0)
        return lidars

    def get_values(self, idx, folder, past_idxs):
        data = self.data[folder]['values']

        current_frame = data[idx]
        x, y, yaw = current_frame[[0,1,2]]

        current_frame = np.reshape(current_frame,(1,-1))
        past_frames = data[idx - np.array(past_idxs),:]
        values = np.concatenate((current_frame, past_frames))
        values = values[:, 0:12] # Remove category column if present
        values = np.nan_to_num(values)
        w_coord = np.transpose(values[:,[0,1]])

        r_coord = util.world_to_relative(x, y, yaw, w_coord)
        values[:,[0,1]] = r_coord.transpose()
        values = np.delete(values,2,1)

        if self.only_lidar: # set all values to 0
            values[:,:] = 0
        elif self.no_intention: # set all intentions to 0
            values[:,(5,6)] = 0

        return values

    def get_output(self, idx, folder, future_idxs):
        data = self.data[folder]['values']

        current_frame = data[idx]
        x, y, yaw = current_frame[[0,1,2]]

        values = data[idx + np.array(future_idxs), :]
        values = values[:, [0,1]] # Keep only x and y
        values = np.nan_to_num(values)
        w_coord = np.transpose(values)

        r_coord = util.world_to_relative(x, y, yaw, w_coord)

        # Return relative coordinates as a 1xSteps array (e.g. [x0,x1,x2,y0,y1,y2])
        return r_coord.reshape(-1)

class RNNDataset(Dataset):

    def __init__(self, dic, no_intention, bptt = 1, frame_stride = 1, transform=None):
        self.bptt = bptt
        self.chunk = bptt*frame_stride
        self.frame_stride = frame_stride

        indices = dic.get('indices')
        indices = np.split(indices[:-(len(indices)%self.chunk)],len(indices)//self.chunk)
        indices = np.array(indices)
        indices = indices[:,0:self.chunk:frame_stride,:]
        self.indices = indices

        lidars = dic.get('lidar')
        lidars = np.split(lidars[:-(len(lidars)%self.chunk)],len(lidars)//self.chunk)
        lidars = np.array(lidars)
        lidars = lidars[:,0:self.chunk:frame_stride,:]
        self.lidars = lidars

        values = dic.get('values')
        values = np.split(values[:-(len(values)%self.chunk)],len(values)//self.chunk)
        values = np.array(values)
        values = values[:,0:self.chunk:frame_stride]
        self.values = values

        outputs = dic.get('output')
        outputs = np.split(outputs[:-(len(outputs)%self.chunk)],len(outputs)//self.chunk)
        outputs = np.array(outputs)
        outputs = outputs[:,0:self.chunk:frame_stride]
        self.outputs = outputs

        self.no_intention = no_intention
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        lidar = []
        for l in self.lidars[idx]:
            temp = np.genfromtxt(l[0], delimiter=',')
            lidar.append(temp)

        values = []
        for v in self.values[idx]:
            temp = np.genfromtxt(v, delimiter=',', skip_header=True)
            if self.no_intention: # set all intentions to 0
                temp[:,(5,6)] = 0
            values.append(temp[0]) #pick only current value
        values = np.nan_to_num(values)

        output = []
        for out in self.outputs[idx]:
            temp = np.genfromtxt(out, delimiter=',', skip_header=True)
            temp = temp.reshape(60)
            output.append(temp)

        foldername_search= re.search('(?<=dataset\/)\w*\/\w*(?=\/)', str(self.lidars[idx])).group()
        foldername = re.sub('\/','_',foldername_search)

        return {'indices': self.indices[idx], 'lidar': np.stack(lidar, axis=0), \
                'value': values, 'output': np.array(output), 'foldername': foldername}
