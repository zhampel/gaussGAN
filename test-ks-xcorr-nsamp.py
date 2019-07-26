from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm as cm

    from scipy import stats

    import pickle

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from torchvision import datasets
    
    from gaussgan.models import Generator
    from gaussgan.datasets import GaussDataset
    from gaussgan.utils import save_model, calc_gradient_penalty, Sampler, enorm, sample_z
    from gaussgan.plots import plot_corr
except ImportError as e:
    print(e)
    raise ImportError

    
Tensor = torch.cuda.FloatTensor


def main():
    global args
    parser = argparse.ArgumentParser(description="GAN sample-xcorr testing script")
    parser.add_argument("-r", "--run_list", dest="run_list", nargs="+", help="List of runs")
    args = parser.parse_args()

    # Get directories as run list
    run_list = args.run_list

    # Pass through run directories
    for irun_dir in run_list:

        # Read in training information from run
        train_deets = '%s/training_details.pkl'%(irun_dir)
        ifile = open(train_deets, 'rb')
        irun_df = pickle.load(ifile)
        ifile.close()

        # Get data, latent dims and scale value
        dim = irun_df['dim']
        latent_dim = irun_df['latent_dim']
        latent_sigma = irun_df['latent_sigma']
        dscale = irun_df['scale_factor']
        data_file = irun_df['data_file']

        # Access saved dataset
        dataset = GaussDataset(file_name=data_file)
        dist_name = dataset.dist_name
        mu = dataset.mu
        sigma = dataset.sigma
        xlo = dataset.xlo
        xhi = dataset.xhi
        
        # Load generator model
        gen_file = "%s/models/generator.pth.tar"%irun_dir
        print("Loading model file: %s"%gen_file)
        generator = Generator(latent_dim=latent_dim, x_dim=dim, dscale=dscale)
        generator.load_state_dict(torch.load(gen_file))
        generator.cuda()
        generator.eval()

        # Define list of testing sample sizes
        list_size = 20
        nsamples_list = np.logspace(1, 4, list_size, dtype=int)
        fine_nsamples = np.logspace(0, 5, 1000)

        # Instantiate empty lists for results
        dval_tru_the = []
        dval_gen_tru = []
        fit_tru_sigmas = []
        fit_gen_sigmas = []
       
        # Run over sample list
        for nsamples in nsamples_list:
            print("Testing with sample size of %i"%nsamples)
    
            # Theoretical dataset
            chi2_rng = np.random.chisquare(dim, nsamples)
            chi2_sqrt = np.sqrt(chi2_rng)
        
            # Test dataset sample
            test_sampler = Sampler(dist_name=dist_name,
                                   dim=dim, n_samples=nsamples, 
                                   mu=mu, sigma=sigma, xlo=xlo, xhi=xhi)
            test_data = test_sampler.sample()
            test_data_numpy = test_data.cpu().data.numpy()
            r_test_samps = enorm(test_data)
    
            # K-S Test Parent to Theoretical
            dval, pval = stats.ks_2samp(r_test_samps, chi2_sqrt)
            dval_tru_the.append(dval)


            # Plot generated data correlation matrix
            test_corr = np.corrcoef(test_data_numpy, rowvar=False)
            figname = '%s/models/test_corr.png'%(irun_dir)
            test_corr_hist, test_fit_sigma = plot_corr(test_corr, figname, title='$X^{t}$ Correlation')

            # Get latent sample vector
            z_samp = sample_z(samples=nsamples, dims=latent_dim, mu=0.0, sigma=latent_sigma)
            z_samp_numpy = z_samp.cpu().data.numpy()[0]

            # Generate a sample
            gen_samp = generator(z_samp)
            gen_data_numpy = gen_samp.cpu().data.numpy() #[0]
            # Euclidean norm calc
            r_gen_samps = enorm(gen_samp)
        
        
            # K-S test test btw test data and generated samples (r distribution)
            dval, pval = stats.ks_2samp(r_test_samps, r_gen_samps)
            dval_gen_tru.append(dval)
           
            # Calculate cross correlations
            gen_corr = np.corrcoef(gen_data_numpy, rowvar=False)
            figname = '%s/models/gen_corr_nsamp%05i.png'%(irun_dir, nsamples)
            ctitle = '$X^{g}$ Correlation for Sample Size $N=%i$'%nsamples
            gen_corr_hist, gen_fit_sigma = plot_corr(corr=gen_corr,
                                                     figname=figname,
                                                     title=ctitle,
                                                     comp_hist=test_corr_hist)

            # Save the fit distribution widths
            fit_gen_sigmas.append(gen_fit_sigma)
            fit_tru_sigmas.append(test_fit_sigma)


        # Convert lists to arrays for plotting/ratio calcs
        dval_tru_the = np.asarray(dval_tru_the)
        dval_gen_tru = np.asarray(dval_gen_tru)
        fit_gen_sigmas = np.asarray(fit_gen_sigmas)
        fit_tru_sigmas = np.asarray(fit_tru_sigmas)

        # Plot results of KSd tests 
        figname='%s/models/samples_KSd.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        axs = fig.add_subplot(111)
        # KS_d values
        axs.plot(nsamples_list, dval_tru_the, color='b', marker='^', markersize=6, markeredgewidth=1, mfc='none', linewidth=0, label='Parent/Theory')
        axs.plot(nsamples_list, dval_gen_tru, color='b', marker='v', markersize=6, linewidth=0, label='Generator/Parent')
        axs.plot(fine_nsamples, 1./np.sqrt(fine_nsamples), 'b--', linewidth=1, alpha=0.5, label=r'$N^{-1/2}$')
        axs.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$', fontsize=16, color='b')
        axs.set_xlabel(r'Sample Size, $N$')
        axs.set_title(r'$\mathrm{KS}$ Distance vs. Sample Size $(n=%i,\ d=%i,\ f=%i)$'%(irun_df['dim'], irun_df['latent_dim'], irun_df['scale_factor']))
        axs.set_xlim(0.75*float(min(nsamples_list)), 1.25*float(max(nsamples_list)))
        axs.set_ylim(1e-3, 5)
        axs.grid()
        axs.tick_params('y', colors='b')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='upper left', ncol=2)

        # Sigma Ratio
        sig_ratio = dval_gen_tru / dval_tru_the
        axr = axs.twinx()
        axr.plot(nsamples_list, sig_ratio, color='r', marker='o', markersize=6, linewidth=0, label='Ratio')
        axr.plot(fine_nsamples, np.ones(len(fine_nsamples)), 'r--', linewidth=1, alpha=0.5, label='1')
        #axr.set_ylabel(r'$\mathrm{KS}^{g/t}_{\mathrm{D}} / \mathrm{KS}^{t/T}_{\mathrm{D}}$', fontsize=16, color='b')
        axr.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$ Ratio', fontsize=16, color='r')
        axr.set_ylim(0., 20.)
        axr.tick_params('y', colors='r')

        fig.tight_layout()
        print(figname)
        fig.savefig(figname)


        # Plot results of xcorr fits 
        figname='%s/models/samples_sigmaxcorr.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        axs = fig.add_subplot(111)
        # Sigma values
        axs.plot(nsamples_list, fit_tru_sigmas, color='b', marker='^', markersize=6, markeredgewidth=1, mfc='none', linewidth=0, label='Parent')
        axs.plot(nsamples_list, fit_gen_sigmas, color='b', marker='v', markersize=6, linewidth=0, label='Generator')
        axs.plot(fine_nsamples, 1./np.sqrt(fine_nsamples), 'b--', linewidth=1, alpha=0.5, label=r'$N^{-1/2}$')
        axs.set_ylabel(r'$\sigma^{t,g}_{\rho}$', fontsize=16, color='b')
        axs.set_xlabel(r'Sample Size, $N$')
        axs.set_title(r'Cross Corr. Width vs. Sample Size $(n=%i,\ d=%i,\ f=%i)$'%(irun_df['dim'], irun_df['latent_dim'], irun_df['scale_factor']))
        axs.set_xlim(0.75*float(min(nsamples_list)), 1.25*float(max(nsamples_list)))
        #axs.set_ylim(0.75*float(min(fit_tru_sigmas)), 1.25*float(max(fit_tru_sigmas)))
        axs.set_ylim(1e-3, 5)
        axs.grid()
        axs.tick_params('y', colors='b')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='upper left', ncol=2)

        # Sigma Ratio
        sig_ratio = fit_gen_sigmas/fit_tru_sigmas
        axr = axs.twinx()
        axr.plot(nsamples_list, sig_ratio, color='r', marker='o', markersize=6, linewidth=0, label='Ratio')
        axr.plot(fine_nsamples, np.ones(len(fine_nsamples)), 'r--', linewidth=1, alpha=0.5, label='1')
        axr.set_ylabel(r'$\sigma^{g}_{\rho} / \sigma^{t}_{\rho}$', fontsize=16, color='r')
        axr.set_ylim(0., 25.)
        axr.tick_params('y', colors='r')

        fig.tight_layout()
        print(figname)
        fig.savefig(figname)


if __name__ == "__main__":
    main()
