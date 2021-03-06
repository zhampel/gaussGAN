from __future__ import print_function

try:
    import numpy as np
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    from itertools import chain as ichain

    from gaussgan.utils import tlog, softmax, initialize_weights, calc_gradient_penalty
except ImportError as e:
    print(e)
    raise ImportError

import math
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )


class Generator(nn.Module):
    """
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    def __init__(self, latent_dim, x_dim, dscale=1, verbose=False):
        super(Generator, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.x_dim = x_dim
        self.verbose = verbose
        self.dscale = dscale
        self.scaled_x_lat = int(dscale*self.latent_dim)
        self.scaled_x_dim = int(dscale*self.x_dim)
        
        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.latent_dim, self.scaled_x_lat),
            #torch.nn.Dropout(0.2),
            nn.BatchNorm1d(self.scaled_x_lat),
            ##torch.nn.Linear(self.latent_dim, 512),
            ##nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
            #GELU(),
            #nn.Sigmoid(),
            #nn.ReLU(),

            #torch.nn.Linear(self.scaled_x_lat, 512),
            #nn.BatchNorm1d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.scaled_x_lat, self.scaled_x_lat),
            #torch.nn.Dropout(0.2),
            nn.BatchNorm1d(self.scaled_x_lat),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
            #GELU(),
            #nn.ReLU(),
            
            #torch.nn.Linear(512, self.scaled_x_dim),
            torch.nn.Linear(self.scaled_x_lat, self.scaled_x_dim),
            #torch.nn.Dropout(0.2),
            nn.BatchNorm1d(self.scaled_x_dim),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
            #GELU(),
            #nn.ReLU(),
            
            torch.nn.Linear(self.scaled_x_dim, x_dim),
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)
    
    def forward(self, z):
        #z = z.unsqueeze(2).unsqueeze(3)
        x_gen = self.model(z)
        # Reshape for output
        #x_gen = x_gen.view(x_gen.size(0), *self.x_dim)
        return x_gen


class Discriminator(nn.Module):
    """
    Input is tuple (X) of an image vector.
    Output is a 1-dimensional value
    """            
    def __init__(self, dim, dscale=1, wass_metric=False, verbose=False):
        super(Discriminator, self).__init__()
        
        self.name = 'discriminator'
        self.wass = wass_metric
        self.dim = dim
        self.verbose = verbose
        self.dscale = dscale
        self.scaled_x_dim = int(dscale*self.dim)
        
        self.model = nn.Sequential(
            nn.Linear(self.dim, self.scaled_x_dim),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
            #GELU(),
            #nn.ReLU(),
            #torch.nn.Dropout(0.2),
            nn.Linear(self.scaled_x_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
            #GELU(),
            #nn.ReLU(),
            #torch.nn.Dropout(0.2),
            #nn.Linear(self.scaled_x_dim, 1024),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(1024, 512),
            #nn.LeakyReLU(0.2, inplace=True),
            
            # Fully connected layers
            torch.nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
            #GELU(),
            #nn.ReLU(),
            #torch.nn.Dropout(0.2),
            
            # Fully connected layers
            torch.nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
            #GELU(),
            #nn.ReLU(),
            #torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
        )
        
        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        validity = self.model(img)
        return validity
