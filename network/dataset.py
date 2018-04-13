import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch

class OurDataset(Dataset):

    def __init__(self, dic, transform=None):

        indices = dic.get('indices')
        self.indices = indices

        lidars = dic.get('lidar')
        self.lidars =  lidars

        values = dic.get('values')
        self.values = values

        outputs = dic.get('output')
        self.outputs = outputs

        self.transform = transform

    def __len__(self):
        return len(self.indices)


#    def __getitem__(self, idx):
#        lidar = np.genfromtxt(self.lidars[idx], delimiter=',')
#        return {'indices': self.indices[idx], 'lidar': lidar, \
#                'value': self.values[idx], 'output': self.outputs[idx]}

    def __getitem__(self, idx):
        lidar = []
        for l in lidars:
            temp = np.genfromtxt(self.lidar[idx], delimiter=',')
            lidar.append(temp)

        values = np.genfromtxt(self.values[idx], delimiter=',', skip_header=True)
        values = np.nan_to_num(values)
        output = np.genfromtxt(self.outputs[idx], delimiter=',', skip_header=True)
        return {'indices': self.indices[idx], 'lidar': lidar, \
                'value': values, 'output': output}
