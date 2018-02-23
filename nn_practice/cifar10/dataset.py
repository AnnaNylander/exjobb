import pandas as pd
import numpy
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch

class CifarDataSet(Dataset):

    def __init__(self, dic, transform=None):
        D_in = 32*32
        D_out = 10
        images = dic.get(b'data')
        self.images =  numpy.reshape(images, (-1, 3, 32, 32))

        labels = dic.get(b'labels')
        one_hots = numpy.zeros([len(labels),D_out])
        for n in range(len(labels)):
            one_hots[n][labels[n]] = 1
        self.labels = one_hots

        # filenames not used.

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'image': self.images[idx], 'label': self.labels[idx]}
