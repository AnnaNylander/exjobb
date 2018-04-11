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

    def __getitem__(self, idx):
        lidar = np.genfromtxt(self.lidars[idx], delimiter=',')
        return {'indices': self.indices[idx], 'lidar': lidar, \
                'value': self.values[idx], 'output': self.outputs[idx]}