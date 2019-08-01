from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from scipy import stats

    import tables

    import pickle
    
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

    from gaussgan.definitions import RUNS_DIR
    from gaussgan.datasets import get_dataloader, GaussDataset
    from gaussgan.models import Generator, Discriminator
    from gaussgan.utils import save_model, calc_gradient_penalty, Sampler, enorm, sample_z, corr2_coeff
    from gaussgan.plots import plot_train_loss, plot_train_curves, compare_histograms, plot_corr, plot_component_ks
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="GAN training script")
    parser.add_argument("-f", "--data_file", dest="data_file", required=True, help="File name of dataset")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=20, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-d", "--latent_dim", dest="latent_dim", default=30, type=int, help="Dimension of latent space")
    parser.add_argument("-s", "--latent_sigma", dest="latent_sigma", default=0.01, type=float, help="Sigma of latent space")
    parser.add_argument("-c", "--dscale", dest="dscale", default=10, type=int, help="Model scaling parameter")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true', help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("-r", "-–num_workers", dest="num_workers", default=1, type=int, help="Number of dataset workers")
    parser.add_argument("-p", "-–postfix", dest="postfix", default="", help="Suffix run name for directory")
    args = parser.parse_args()

    data_file = args.data_file
    device_id = args.gpu
    num_workers = args.num_workers

    # Access saved dataset
    dataset = GaussDataset(file_name=data_file)
    dist_name = dataset.dist_name
    dim = dataset.dim
    mu = dataset.mu
    sigma = dataset.sigma
    xlo = dataset.xlo
    xhi = dataset.xhi
    print("Getting dataset from %s"%data_file)
    print("Dataset size: ", dataset.__len__())

    # Latent space info
    latent_dim = args.latent_dim
    latent_sigma = args.latent_sigma
    dscale = args.dscale
   
    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 5
    #n_skip_iter = 1

    # Wasserstein metric flag
    wass_metric = args.wass_metric
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'

    # Make directory structure for this run
    sep_und = '_'
    run_name_comps = [dist_name, 'dim%s'%str(dim), 'c%s'%str(dscale), 'z%s'%str(latent_dim), 's%s'%str(latent_sigma), mtype, 'bs%i'%batch_size]
    run_name = sep_und.join(run_name_comps)

    # Run name suffix
    postfix = args.postfix
    if not postfix == "":
        run_name = os.path.join(run_name, sep_und, postfix)

    run_dir = os.path.join(RUNS_DIR, run_name)
    samples_dir = os.path.join(run_dir, 'samples')
    models_dir = os.path.join(run_dir, 'models')
    training_details_file = '%s/training_details.pkl'%(run_dir)
    if os.path.isfile(training_details_file):
        print('Training already performed. Please move or delete %s to re-run. Exiting...'%run_dir)
        import sys
        sys.exit()

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))
   
    x_shape = dim
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    generator = Generator(latent_dim, x_shape, dscale)
    discriminator = Discriminator(dim=dim, wass_metric=wass_metric)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Configure training data loader
    dataloader = get_dataloader(dataset=dataset,
                                batch_size=batch_size,
                                train_set=True,
                                num_workers=num_workers)
    
    # Sample from training set
    n_train_samples = 100
    train_dataloader = get_dataloader(dataset=dataset,
                                      batch_size=n_train_samples,
                                      train_set=True,
                                      num_workers=num_workers)
    train_data = next(iter(train_dataloader))
    train_data_numpy = train_data.cpu().data.numpy()
    r_train = enorm(train_data)

    # Plot generated data correlation matrix
    train_corr = np.corrcoef(train_data_numpy, rowvar=False)
    figname = '%s/train_corr.png'%(run_dir)
    train_corr_hist, train_fit_sigma = plot_corr(train_corr, figname, title='$X^{train}$ Correlation')


    # Test dataset (out of training set)
    n_test_samples = n_train_samples #1000
    test_sampler = Sampler(dist_name=dist_name,
                           dim=dim, n_samples=n_test_samples, 
                           mu=mu, sigma=sigma, xlo=xlo, xhi=xhi)
    test_data = test_sampler.sample()
    test_data_numpy = test_data.cpu().data.numpy()
    r_test = enorm(test_data)

    # Plot generated data correlation matrix
    test_corr = np.corrcoef(test_data_numpy, rowvar=False)
    figname = '%s/test_corr.png'%(run_dir)
    test_corr_hist, test_fit_sigma = plot_corr(test_corr, figname, title='$X^{test}$ Correlation')

    # Prepare test set component histograms
    test_hist_list = [None] * dim
    xemin = np.floor(np.min(test_data_numpy))
    xemax = np.ceil(np.max(test_data_numpy))
    rmax = 1.2*int(np.max(r_test))

    # Squeeze limits for Cauchy distribution visualization
    if 'cauchy' in dist_name:
        cstd = np.std(test_data_numpy)
        xemin = -0.1*cstd
        xemax = 0.1*cstd
        rmax *= 0.005

    # Histogram bin defs
    xedges = np.linspace(xemin, xemax, 50)
    xcents = (xedges[1:]-xedges[:-1])/2 + xedges[0:-1]
    redges = np.linspace(0, rmax, 50)
    rcents = (redges[1:]-redges[:-1])/2 + redges[0:-1]
    
    yhistmax = -1.0
    for idim in range(dim):
        # Distribution for each dimension component
        xhist = np.histogram(test_data_numpy[:, idim], bins=xedges)[0]
        xhist = xhist / np.sum(xhist)
        test_hist_list[idim] = xhist
        yhistmax = max(yhistmax, np.max(xhist))

    # Theoretical dataset
    chi2_rng = np.random.chisquare(dim, n_test_samples)
    chi2_sqrt = np.sqrt(chi2_rng)

    # K-S Test
    dval, pval = stats.ks_2samp(r_test, chi2_sqrt)
    print("Comparing theoretical chi2 dist (sqrt) to sampled distribution:")
    print("P-Value: %.04f\tDist-Value: %.04f"%(dval, pval))

    # Euclidean norm distributions
    train_hist, _ = np.histogram(r_train, bins=redges)
    test_hist, _ = np.histogram(r_test, bins=redges)
    chi_hist, _ = np.histogram(chi2_sqrt, bins=redges)
    train_hist = train_hist / float(n_train_samples)
    test_hist = test_hist / float(n_test_samples)
    chi_hist = chi_hist / float(n_test_samples)
   
    #optimizer_G = torch.optim.SGD(generator.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
    #optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=lr, momentum=0.9)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

    # ----------
    #  Training
    # ----------
    g_l = []
    d_l = []
    dval_train_i = []
    pval_train_i = []
    dval_test_i = []
    pval_test_i = []
    gr_fit_sigma_list = []
    gt_fit_sigma_list = []
    gen_fit_sigma_list = []
    disc_g_mean = []
    disc_r_mean = []
    
    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
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
            z_latent = sample_z(samples=real_samples.size()[0], dims=latent_dim, mu=0.0, sigma=latent_sigma)
            #z_latent = sample_zu(samples=real_samples.size()[0], dims=latent_dim)

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
                    g_loss = torch.mean(D_gen)
                else:
                    # Vanilla GAN loss
                    valid = Variable(Tensor(gen_samples.size(0), 1).fill_(1.0), requires_grad=False)
                    g_loss = bce_loss(D_gen, valid)
    
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
                fake = Variable(Tensor(gen_samples.size(0), 1).fill_(0.0), requires_grad=False)
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2
    
            d_loss.backward()
            optimizer_D.step()


            # Save training losses
            d_l.append(d_loss.item())
            g_l.append(g_loss.item())

            # Save mean of discriminator outputs
            disc_r_mean.append(D_real.mean().item())
            disc_g_mean.append(1-D_gen.mean().item())
   
        # Generator in eval mode
        generator.eval()

        # Set number of examples for cycle calcs
        n_samp = n_test_samples

        # Generate sample instances
        #z_samp = sample_zu(samples=n_samp, dims=latent_dim)
        z_samp = sample_z(samples=n_samp, dims=latent_dim, mu=0.0, sigma=latent_sigma)
        gen_samples_samp = generator(z_samp)


        # Compare true/gen distributions for each component
        gen_data_numpy = gen_samples_samp.cpu().data.numpy()

        ks_d_list = []
        ks_p_list = []

        figname = '%s/comp_hist_epoch%05i.png'%(samples_dir, epoch)
        fig = plt.figure(figsize=(18,12))
        # For large dimensionalities, cull for visualization
        subdim = np.min([100, dim])
        pdims = np.ceil(np.sqrt(subdim))
        for idim in range(subdim):
            # Initialize histogram for each dimension component
            xhist = np.histogram(gen_data_numpy[:, idim], bins=xedges)[0]
            xhist = xhist / np.sum(xhist)
            iax = fig.add_subplot(pdims, pdims, idim + 1)
            iax.step(xcents, test_hist_list[idim], linewidth=1.5, c='k')
            iax.step(xcents, xhist, linewidth=1.5, c='r')
            iax.grid()
            iax.set_xlim(xemin, xemax)
            iax.set_ylim(0.0, yhistmax*1.5)
            #plt.yscale('log')
            plt.axis('off')
            
       
        # Label color scheme in title at bottom of figure
        fig.text(0.5, 0.03, 'Epoch %i'%epoch, ha='center', fontsize=16, color='k')
        fig.text(0.45, 0.0125, 'Generated', ha='left', fontsize=16, color='r')
        tactext = 'True &                  Components'
        fig.text(0.5, 0.0125, tactext, ha='center', fontsize=16, color='k')

        plt.tight_layout()
        fig.savefig(figname)

        # Perform KS test for individual components
        for idim in range(dim):
            dval, pval = stats.ks_2samp(gen_data_numpy[:, idim], test_data_numpy[:, idim])
            ks_d_list.append(dval)
            ks_p_list.append(pval)
       

        # Hack for step plot
        ks_d_list.append(ks_d_list[-1])
        ks_p_list.append(ks_p_list[-1])

        # Save results of KS test to figure
        figname = '%s/ks_epoch%05i.png'%(samples_dir, epoch)
        plot_component_ks(figname=figname, epoch=epoch, dim=dim,
                          ks_d_list=ks_d_list,
                          ks_p_list=ks_p_list
                         )


        # Plot generated data correlation matrix
        gen_corr = np.corrcoef(gen_data_numpy, rowvar=False)
        figname = '%s/corr_gen_epoch%05i.png'%(samples_dir, epoch)
        ctitle = '$X^{g}$ Correlation at Epoch %i'%epoch
        gen_corr_hist, gen_fit_sigma = plot_corr(corr=gen_corr,
                                                 figname=figname,
                                                 title=ctitle,
                                                 comp_hist=test_corr_hist)

        gen_fit_sigma_list.append(gen_fit_sigma)

        
        # Plot generated data - train data correlation matrix
        #gr_corr = corr2_coeff(gen_data_numpy.T, gen_data_numpy.T)
        gr_corr = corr2_coeff(gen_data_numpy.T, train_data_numpy.T)
        figname = '%s/corr_train_epoch%05i.png'%(samples_dir, epoch)
        ctitle = '$X^{g}, X^{r}$ Correlation at Epoch %i'%epoch
        gr_corr_hist, gr_fit_sigma = plot_corr(corr=gr_corr,
                                               figname=figname,
                                               title=ctitle,
                                               comp_hist=train_corr_hist)

        gr_fit_sigma_list.append(gr_fit_sigma)


        # Plot generated data - test data correlation matrix
        gt_corr = corr2_coeff(gen_data_numpy.T, test_data_numpy.T)
        figname = '%s/corr_test_epoch%05i.png'%(samples_dir, epoch)
        ctitle = '$X^{g}, X^{t}$ Correlation at Epoch %i'%epoch
        gt_corr_hist, gt_fit_sigma = plot_corr(corr=gt_corr,
                                               figname=figname,
                                               title=ctitle,
                                               comp_hist=test_corr_hist)

        gt_fit_sigma_list.append(gt_fit_sigma)

        
        # Euclidean norm calc and comparison
        r_gen_samps = enorm(gen_samples_samp)
        # Bin samples into normalized histogram
        gen_hist, _ = np.histogram(r_gen_samps, bins=redges)
        gen_hist = gen_hist / float(n_samp)

        # Plot norm distributions
        figname = '%s/rhist_epoch%05i.png'%(samples_dir, epoch)
        compare_histograms(hist_list=[test_hist, gen_hist],
                           centers=[rcents, rcents],
                           labels=['Parent', 'Generated', '%i'%epoch],
                           ylims=[0.0005, 1.0, True],
                           figname=figname)
      
        # K-S test test btw test/train data and generated samples (r distribution)
        dval_test, pval_test = stats.ks_2samp(r_test, r_gen_samps)
        dval_train, pval_train = stats.ks_2samp(r_train, r_gen_samps)
        
        dval_test_i.append(dval_test)
        pval_test_i.append(pval_test)
        dval_train_i.append(dval_train)
        pval_train_i.append(pval_train)

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [G: %f] [p-val: %.02e]" % (epoch, 
                                                                   n_epochs, 
                                                                   d_loss.item(),
                                                                   g_loss.item(),
                                                                   pval)
              )
 

    # Save training results
    train_df = {
                'data_file' : data_file,
                'dim' : dim,
                'n_epochs' : n_epochs,
                'batch_size' : batch_size,
                'learning_rate' : lr,
                'beta_1' : b1,
                'beta_2' : b2,
                'weight_decay' : decay,
                'n_skip_iter' : n_skip_iter,
                'latent_dim' : latent_dim,
                'latent_sigma' : latent_sigma,
                'wass_metric' : wass_metric,
                'scale_factor' : dscale,
                'gen_loss' : ['G(z)', g_l],
                'disc_loss' : ['D(x)', d_l],
                'disc_r_mean' : ['D(x)', disc_r_mean],
                'disc_g_mean' : ['1-D(G(z))', disc_g_mean],
                'pvalues_test' : ['pvals', pval_test_i],
                'dvalues_test' : ['dvals', dval_test_i],
                'pvalues_train' : ['pvals', pval_train_i],
                'dvalues_train' : ['dvals', dval_train_i],
                'true_corr' : ['test_corr', test_corr],
                'gen_corr' : ['gen_corr', gen_corr],
                'train_xcorr' : ['train_corr_hist', train_corr_hist],
                'true_xcorr' : ['test_corr_hist', test_corr_hist],
                'gen_xcorr' : ['gen_corr_hist', gen_corr_hist],
                'gr_xcorr' : ['gr_corr_hist', gr_corr_hist],
                'gt_xcorr' : ['gt_corr_hist', gt_corr_hist],
                'n_train_samples' : n_train_samples,
                'n_test_samples' : n_test_samples,
                'test_fit_sigma' : test_fit_sigma,
                'train_fit_sigma' : train_fit_sigma,
                'gen_fit_sigma' : ['gen_sigma' , gen_fit_sigma_list],
                'gr_fit_sigma' : ['gr_sigma' , gr_fit_sigma_list],
                'gt_fit_sigma' : ['gt_sigma' , gt_fit_sigma_list],
               }

    fjson = open(training_details_file, 'wb')
    pickle.dump(train_df, fjson)
    fjson.close()


    # Save current state of trained models
    model_list = [discriminator, generator]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()
