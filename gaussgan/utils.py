from __future__ import print_function

try:
    import os
    import numpy as np
    from scipy.stats import truncnorm as truncnorm
    from scipy.stats import cauchy as cauchy
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    from itertools import chain as ichain

except ImportError as e:
    print(e)
    raise ImportError


# Nan-avoiding logarithm
def tlog(x):
      return torch.log(x + 1e-8)


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


# Gaussian function
def gaussian(x, mu, sig):
    """
    Gaussian function
    """
    #return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / np.sqrt(2*np.pi) / sig
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# Euclidean norm
def enorm(data=None, cpu=True):
    """
    Provided a batch of vectors,
    return Euclidean norms
    """
    # Magnitude squared of vectors
    r2 = torch.sum(data * data, dim=1)
    # Vector magnitudes
    r = torch.sqrt(r2)

    # Return numpy array as default
    if cpu:
        r = r.cpu().data.numpy()

    return r


# Save a provided model to file
def save_model(models=[], out_dir=''):

    # Ensure at least one model to save
    assert len(models) > 0, "Must have at least one model to save."

    # Save models to directory out_dir
    for model in models:
        filename = model.name + '.pth.tar'
        outfile = os.path.join(out_dir, filename)
        torch.save(model.state_dict(), outfile)


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def sample_trunc_z(samples=64, dims=10, mu=0.0, sigma=1.0, xlo=-2.0, xhi=2.0, req_grad=False):

    Tensor = torch.cuda.FloatTensor
    
    ## Sample noise as generator input, zn
    z_trunc = []
    for idim in range(samples):
        z_trunc.append(truncnorm.rvs(xlo, xhi, size=dims)*sigma + mu)

    z_trunc = np.asarray(z_trunc)

    #isize = (samples, dims)
    #u1 = torch.rand(isize) * (1 - np.exp(-2)) + np.exp(-2)
    #u2 = torch.rand(isize)
    #z = torch.sqrt(-2 * np.log(u1)) * torch.cos(2*np.pi*u2)

    z = Variable(Tensor(z_trunc), requires_grad=req_grad)

    # Return components of latent space variable
    return z


# Sample a random latent space vector
def sample_z(samples=64, dims=10, mu=0.0, sigma=1.0, req_grad=False):

    Tensor = torch.cuda.FloatTensor

    # Sample noise as generator input, zn
    #z = Variable(Tensor(np.random.normal(mu, sigma, (samples, dims))), requires_grad=req_grad)

    # Check if mu, sigma are scalars
    if np.isscalar(mu):
        mu = mu*np.ones(dims)
    if np.isscalar(sigma):
        sigma = np.zeros((dims, dims))+np.diag(sigma*np.ones(dims))

    assert dims == len(mu), \
           "Mean vector must have length {} equal to data dimensions {}.".format(len(mu), dims)
    assert dims == sigma.shape[0] and dims == sigma.shape[1], \
           "Covariance matrix must have square shape {}x{} equal to data dimensions {}.".format(sigma.shape[0], sigma.shape[1], dims)
        

    # Generate multivariate normal sample batch
    z_np = np.random.multivariate_normal(mu, sigma, (samples))
    z = Variable(Tensor(z_np), requires_grad=req_grad)

    # Return components
    return z


# Sample from a Caucy distribution
def sample_cauchy(samples=54, dims=10, loc=0.0, scale=1.0, req_grad=False):

    Tensor = torch.cuda.FloatTensor
    
    z_cauchy = []
    for idim in range(samples):
        z_cauchy.append(cauchy.rvs(loc=loc, scale=scale, size=dims))

    z_cauchy = np.asarray(z_cauchy)

    z = Variable(Tensor(z_cauchy), requires_grad=req_grad)

    # Return components of latent space variable
    return z


# Sample a random latent space vector
def sample_zu(samples=64, dims=10, xlo=0.0, xhi=1.0, req_grad=False):

    Tensor = torch.cuda.FloatTensor
    
    # Sample noise as generator input, zn
    z = Variable(Tensor(np.random.uniform(low=xlo, high=xhi, size=(samples, dims))), requires_grad=req_grad) 
    #z = Variable(Tensor(np.random.random((samples, dims))), requires_grad=req_grad) 
    #z = Variable(Tensor(np.random.normal(mu, sigma, (samples, dims))), requires_grad=req_grad)

    # Return components of latent space variable
    return z


# Sampling function dictionary
DATASET_FN_DICT = {'gauss' : sample_z,
                   'trunc_gauss' : sample_trunc_z,
                   'cauchy' : sample_cauchy,
                   'uniform' : sample_zu
                  }


sampler_list = DATASET_FN_DICT.keys()


def get_sampler(dist_name='gauss'):
    """
    Convenience function for retrieving
    allowed datasets.
    Parameters
    ----------
    name : {'gauss', 'trunc_gauss', 'uniform'}
          Name of dataset
    Returns
    -------
    fn : function
         sampling function
    """
    if dist_name in DATASET_FN_DICT:
        fn = DATASET_FN_DICT[dist_name]
        return fn
    else:
        raise ValueError('Invalid sampler, {}, entered. Must be '
                         'in {}'.format(dist_name, DATASET_FN_DICT.keys()))


# Sampler class
class Sampler(object):
    """
    Sampler function class
    """
    def __init__(self, dist_name='gauss', dim=1, n_samples=64, mu=0.0, sigma=1.0, xlo=-2.0, xhi=2.0):
        self.dist_name = dist_name
        self.sampler = get_sampler(dist_name=dist_name)
        self.dim = dim
        self.n_samples = n_samples
        self.mu = mu
        self.sigma = sigma
        self.xlo = xlo
        self.xhi = xhi

    # Sample a random latent space vector
    def sample(self):
        if self.dist_name == 'gauss':
            return sample_z(samples=self.n_samples, dims=self.dim, mu=self.mu, sigma=self.sigma, req_grad=False)
        if self.dist_name == 'trunc_gauss':
            return sample_trunc_z(samples=self.n_samples, dims=self.dim, mu=self.mu, sigma=self.sigma, xlo=self.xlo, xhi=self.xhi, req_grad=False)
        if self.dist_name == 'cauchy':
            return sample_cauchy(samples=self.n_samples, dims=self.dim, loc=self.mu, scale=self.sigma, req_grad=False)
        if self.dist_name == 'uniform':
            return sample_zu(samples=self.n_samples, dims=self.dim, req_grad=False)


# Gradient penalty calculation for Wasserstein GAN
def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


def smooth(x, window_len=10, window='flat'):
    """smooth the data using a window with requested size.
    From: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y= np.convolve(w/w.sum(),s,mode='valid')
    return y


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
