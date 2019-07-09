from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm as cm

    import pickle

    from gaussgan.plots import plot_train_loss, plot_train_curves, compare_histograms, plot_corr
    
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="GAN training plotting script")
    parser.add_argument("-r", "--run_list", dest="run_list", nargs="+", help="List of runs")
    args = parser.parse_args()

    # Get directories as run list
    run_list = args.run_list

    # Pass through run directories
    for irun_dir in run_list:
   
        train_deets = '%s/training_details.pkl'%(irun_dir)

        # Skip if training file not found
        if not os.path.isfile(train_deets):
            print("\n\t%s DNE. Skipping...\n"%train_deets)
            continue

        # Read in training information from run
        ifile = open(train_deets, 'rb')
        irun_df = pickle.load(ifile)
        ifile.close()
        # Get data, latent dims and scale value
        dim = irun_df['dim']
        n_epochs = irun_df['n_epochs']
        latent_dim = irun_df['latent_dim']
        dscale = irun_df['scale_factor']
        # Test samples
        n_test_samples = irun_df['n_test_samples']
        # Get KSd values
        dvalues = irun_df['dvalues'][1]
        # Get KSp values
        pvalues = irun_df['pvalues'][1]


        # Get Generated sigma fits
        sigma_theo = irun_df['test_fit_sigma']
        gen_fit_sigma = irun_df['gen_fit_sigma'][1]
        gen_fit_sigma /= sigma_theo

        # Plot some training results
        plot_train_curves(df=irun_df,
                          arr_list=['gen_loss', 'disc_loss', 'disc_r_mean', 'disc_g_mean'],
                          figname='%s/training_model_curves.png'%(irun_dir)
                         )

        plot_train_loss(df=irun_df,
                        arr_list=['gen_loss', 'disc_loss'],
                        figname='%s/training_model_losses.png'%(irun_dir)
                       )


        # Plot results of xcorr fit 
        figname='%s/training_sigmaxcorr.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        ax = fig.add_subplot(111)
        epochs = range(0, n_epochs)
        # D-Values
        ax.plot(epochs, gen_fit_sigma, color='b', marker='o', linewidth=0)
        ax.set_ylabel(r'$\sigma^{g}_{\rho} / \sigma^{t}_{\rho}$', fontsize=16)
        ax.set_xlabel(r'Epoch')
        ax.set_title(r'Cross Corr. Width Ratio for $N=%i$ samples'%n_test_samples)
        fig.tight_layout()
        print(figname)
        fig.savefig(figname)


        # Plot results of KS test 
        figname='%s/training_kstest.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        axd = fig.add_subplot(111)
        epochs = range(0, n_epochs)
        # D-Values
        axd.plot(epochs, dvalues, label=r'$KS_{D}$', color='b', marker='o', linewidth=0)
        axd.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$', color='b')
        axd.tick_params('y', colors='b')
        axd.set_xlabel(r'Epoch')
        axd.set_title(r'KS Test on Euclidean Norm')
        # P-Values
        axp = axd.twinx()
        axp.plot(epochs, pvalues, label=r'$KS_{p}$', color='r', marker='s', linewidth=0)
        axp.set_ylim(0, 1.1*max(pvalues))
        axp.set_ylabel(r'$\mathrm{KS}_{\mathrm{p}}$', color='r')
        axp.tick_params('y', colors='r')
        fig.tight_layout()
        print(figname)
        fig.savefig(figname)

        
        # Plot results of KS test 
        figname='%s/training_ksd_sxcorr.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        axd = fig.add_subplot(111)
        epochs = range(0, n_epochs)
        # D-Values
        axd.plot(epochs, dvalues, label=r'$KS_{D}$', color='b', marker='o', linewidth=0)
        axd.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$', color='b')
        axd.tick_params('y', colors='b')
        axd.set_xlabel(r'Epoch')
        axd.set_title(r'KS Test on Euclidean Norm')
        # P-Values
        axp = axd.twinx()
        axp.plot(epochs, gen_fit_sigma, color='r', marker='s', linewidth=0)
        axp.set_ylabel(r'$\sigma^{g}_{\rho} / \sigma^{t}_{\rho}$', fontsize=16, color='r')
        axp.tick_params('y', colors='r')
        fig.tight_layout()
        print(figname)
        fig.savefig(figname)

    

if __name__ == "__main__":
    main()
