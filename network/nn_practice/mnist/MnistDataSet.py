import pandas as pd
import numpy
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch

class MnistDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv = pd.read_csv(csv_file)

        inputs = csv.iloc[:,:].as_matrix()
        inputs = inputs[:,1:len(inputs)]
        self.images =  numpy.reshape(inputs, (-1, 28, 28))

        outputs = csv.iloc[:,0].as_matrix()
        one_hots = numpy.zeros([len(inputs),10])
        for n in range(len(inputs)):
            one_hots[n][outputs[n]] = 1
        self.labels = one_hots

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'image': self.images[idx], 'label': self.labels[idx]}

        #img_name = os.path.join(self.root_dir,
        #                        self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        #if self.transform:
        #    sample = self.transform(sample)

        #return sample
