import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch

class OurDataset(Dataset):

    def __init__(self, dic, no_intention, only_lidar, transform=None):

        indices = dic.get('indices')
        self.indices = indices

        lidars = dic.get('lidar')
        self.lidars =  lidars

        values = dic.get('values')
        self.values = values

        outputs = dic.get('output')
        self.outputs = outputs

        categories = dic.get('category')
        self.categories = categories

        self.no_intention = no_intention
        self.only_lidar = only_lidar
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        lidar = []
        for l in self.lidars[idx]:
            temp = np.genfromtxt(l, delimiter=',')
            lidar.append(temp)

        values = np.genfromtxt(self.values[idx], delimiter=',', skip_header=True)
        values = np.nan_to_num(values)
        if self.only_lidar: # set all values to 0
            values[:,:] = 0
        elif self.no_intention: # set all intentions to 0
            values[:,(5,6)] = 0
        output = np.genfromtxt(self.outputs[idx], delimiter=',', skip_header=True)

        output = output.reshape(60)
        #print(str(self.lidars[idx]))
        #print(self.indices[idx])
        foldername_search= re.search('(?<=dataset\/)\w*\/\w*\/\w*\/', str(self.lidars[idx])).group()
        #print(foldername_search)
        foldername = re.sub('\/','',foldername_search)
        foldername = re.sub('train|test|validate','',foldername)
        foldername = re.sub('eukaryote','',foldername)

        return {'indices': self.indices[idx],
                'lidar': np.stack(lidar, axis=0),
                'value': values,
                'output': output,
                'foldername': foldername,
                'category': self.categories[idx]}


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
