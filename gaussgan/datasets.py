from __future__ import print_function

try:
    import os
    import numpy as np

    import tables
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
except ImportError as e:
    print(e)
    raise ImportError



class GaussDataset(Dataset):
    """
    Gaussian dataset class
    """
    def __init__(self, file_name='', transform=None):
        """
        Input
        -----
            file_name : string
                        name of file with dataset
            transform : object
                        optional transforms to apply to samples
        """
        self.file_name = file_name
        self.transform = transform
        self.data_file = self.load_file()

    def load_file(self):
        assert os.path.isfile(self.file_name), "Dataset file %s dne. Exiting..."%self.file_name
        data_file = tables.open_file(self.file_name, mode='r')
        return data_file

    def __len__(self):
        return self.data_file.root.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data_file.root.data[idx,:]
        return sample



def get_dataloader(dataset=None, batch_size=64, train_set=True):
    """
    Function to provide a DataLoader given a dataset.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True)

    return dataloader
