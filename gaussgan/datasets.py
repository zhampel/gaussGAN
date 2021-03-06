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
        self.dataset_len = self.load_file()
        self.datatable = None

    def load_file(self):
        assert os.path.isfile(self.file_name), "Dataset file %s dne. Exiting..."%self.file_name
        data_file = tables.open_file(self.file_name, mode='r')
        # Get metadata from file
        meta_data = data_file.root.metadata
        self.dist_name = meta_data.dist[0].decode("utf-8")
        self.dim = meta_data.dim[0]
        self.n_samples = meta_data.n_samples[0]
        self.mu = meta_data.mu[0]
        self.sigma = meta_data.sigma[0]
        self.xlo = meta_data.xlo[0]
        self.xhi = meta_data.xhi[0]
        dataset_len = data_file.root.data.shape[0]
        data_file.close()
        return dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.datatable is None:
            self.datatable = tables.open_file(self.file_name, mode='r')
        return self.datatable.root.data[idx]



def get_dataloader(dataset=None, batch_size=64, train_set=True, num_workers=1):
    """
    Function to provide a DataLoader given a dataset.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True)

    return dataloader
