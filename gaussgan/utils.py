from __future__ import print_function

try:
    import os
    import numpy as np
    
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


# Cross Entropy loss with two vector inputs
def cross_entropy(pred, soft_targets):
    log_softmax_pred = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.mean( torch.sum(- soft_targets * log_softmax_pred, 1) )


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

# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, req_grad=False):

    Tensor = torch.cuda.FloatTensor
    
    # Sample noise as generator input, zn
    zn = Variable(Tensor(np.random.normal(0, 1, (shape, latent_dim))), requires_grad=req_grad)

    # Return components of latent space variable
    return zn


def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
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
