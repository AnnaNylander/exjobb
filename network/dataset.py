import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch

class OurDataset(Dataset):

    def __init__(self, dic, no_intention, transform=None):

        indices = dic.get('indices')
        self.indices = indices

        lidars = dic.get('lidar')
        self.lidars =  lidars

        values = dic.get('values')
        self.values = values

        outputs = dic.get('output')
        self.outputs = outputs

        self.no_intention = no_intention
        self.transform = transform

    def __len__(self):
        return len(self.indices)


#    def __getitem__(self, idx):
#        lidar = np.genfromtxt(self.lidars[idx], delimiter=',')
#        return {'indices': self.indices[idx], 'lidar': lidar, \
#                'value': self.values[idx], 'output': self.outputs[idx]}

    def __getitem__(self, idx):
        lidar = []
        for l in self.lidars[idx]:
            temp = np.genfromtxt(l, delimiter=',')
            lidar.append(temp)

        values = np.genfromtxt(self.values[idx], delimiter=',', skip_header=True)
        values = np.nan_to_num(values)
        if self.no_intention: # set all intentions to 0
            values[:,(5,6)] = 0
        output = np.genfromtxt(self.outputs[idx], delimiter=',', skip_header=True)

        foldername = ''
        foldername_search= re.search('(?<=\/)\w*(?=\/(train|test|validate))', str(self.lidars[idx])) #TODO fix regex so we can save properly
        if foldername_search is not None:
            foldername = foldername_search.group()

        return {'indices': self.indices[idx], 'lidar': np.stack(lidar, axis=0), \
                'value': values, 'output': output, 'foldername': foldername}
