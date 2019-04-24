from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

    import tables

    import pandas as pd
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from itertools import chain as ichain

    from gaussgan.definitions import DATASETS_DIR, RUNS_DIR
    from gaussgan.datasets import get_dataloader, GaussDataset
    from gaussgan.models import Generator, Discriminator
    from gaussgan.utils import tlog, save_model, calc_gradient_penalty, sample_z, cross_entropy
    from gaussgan.plots import plot_train_loss
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="GAN training script")
    #parser.add_argument("-r", "--run_name", dest="run_name", default='gaussgan', help="Name of training run")
    parser.add_argument("-d", "--dim", dest="dimensions", default=1, type=int, help="Number of dimensions")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    #parser.add_argument("-l", "--dim_list", dest="dim_list", nargs='+', default=[1, 2, 3, 10, 100], type=int, help="Number of samples")
    args = parser.parse_args()

    dim = args.dimensions
    run_name = 'dim%i'%dim#args.run_name

    # Make directory structure for this run
    run_dir = os.path.join(RUNS_DIR, run_name)
    data_dir = os.path.join(DATASETS_DIR)
    samples_dir = os.path.join(run_dir, 'samples')
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))
   
    # Access saved dataset
    data_file_name = '%s/data_dim%i.h5'%(data_dir, dim)
    dataset = GaussDataset(file_name=data_file_name)
    print("Getting dataset from %s"%data_file_name)
    print("Dataset size: ", dataset.__len__())
    #atom = tables.Float64Atom()
    #array_c = f.create_earray(f.root, 'data', atom, (0, ROW_SIZE))

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    # Latent space info
    latent_dim = 30
   
    # Wasserstein metric flag
    wass_metric = True
    
    x_shape = dim
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    generator = Generator(latent_dim, x_shape)
    discriminator = Discriminator(dim=dim, wass_metric=wass_metric)
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Configure training data loader
    dataloader = get_dataloader(dataset=dataset, batch_size=batch_size)
   
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

    # ----------
    #  Training
    # ----------
    g_l = []
    d_l = []
    
    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
        #for i, (samples, itruth_label) in enumerate(dataloader):
        for i, samples in enumerate(dataloader):
           
            # Ensure generator is trainable
            generator.train()
            # Zero gradients for models
            generator.zero_grad()
            discriminator.zero_grad()
            
            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # -----------------
            #  Train Generator 
            # -----------------
            
            optimizer_G.zero_grad()
            
            # Sample random latent variables
            z_latent = sample_z(samples=n_samples, dims=dim, mu=0.0, sigma=1.0)

            # Generate a batch of samples
            gen_samples = generator(z_latent)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_samples)
            D_real = discriminator(real_samples)
            
            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % n_skip_iter == 0):
    
                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    g_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
                else:
                    # Vanilla GAN loss
                    g_loss = -torch.mean(tlog(D_gen)) + betan * zn_loss + betac * zc_loss
    
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_samples, gen_samples)

                # Wasserstein GAN loss w/gradient penalty
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
                
            else:
                # Vanilla GAN loss
                d_loss = -torch.mean(tlog(D_real) - tlog(1 - D_gen))
    
            d_loss.backward()
            optimizer_D.step()


        # Save training losses
        d_l.append(d_loss.item())
        g_l.append(g_loss.item())
   
        # Generator in eval mode
        generator.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp



        ## Cycle through randomly sampled encoding -> generator -> encoder
        z_samp sample_z(shape=n_samp latent_dim=latent_dim)
        # Generate sample instances
        gen_samples_samp = generator(z_samp)
      
        # Save cycled and generated examples!
        z = sample_z(samples=n_samples, dims=dim, mu=0.0, sigma=1.0)


        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                     n_epochs, 
                                                     d_loss.item(),
                                                     g_loss.item())
              )
        

    ## Save training results
    #train_df = pd.DataFrame({
    #                         'n_epochs' : n_epochs,
    #                         'learning_rate' : lr,
    #                         'beta_1' : b1,
    #                         'beta_2' : b2,
    #                         'weight_decay' : decay,
    #                         'n_skip_iter' : n_skip_iter,
    #                         'latent_dim' : latent_dim,
    #                         'n_classes' : n_c,
    #                         'beta_n' : betan,
    #                         'beta_c' : betac,
    #                         'wass_metric' : wass_metric,
    #                         'gen_loss' : ['G', g_l],
    #                         'disc_loss' : ['D', d_l],
    #                        })

    #train_df.to_csv('%s/training_details.csv'%(run_dir))


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/training_model_losses.png'%(run_dir)
                    )


    # Save current state of trained models
    model_list = [discriminator, encoder, generator]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()
