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
        # Get KSd test values
        dvalues_test = irun_df['dvalues_test'][1]
        # Get KSp test values
        pvalues_test = irun_df['pvalues_test'][1]
        # Get KSd train values
        dvalues_train = irun_df['dvalues_train'][1]
        # Get KSp train values
        pvalues_train = irun_df['pvalues_train'][1]


        # Get sigma fits
        sigma_theo = irun_df['test_fit_sigma']
        gen_fit_sigma = irun_df['gen_fit_sigma'][1]
        gen_fit_sigma /= sigma_theo
        gr_fit_sigma = irun_df['gr_fit_sigma'][1]
        gr_fit_sigma /= sigma_theo
        gt_fit_sigma = irun_df['gt_fit_sigma'][1]
        gt_fit_sigma /= sigma_theo

        # Plot some training results
        plot_train_curves(df=irun_df,
                          arr_list=['gen_loss', 'disc_loss', 'disc_r_mean', 'disc_g_mean'],
                          figname='%s/training_model_curves.png'%(irun_dir)
                         )

        plot_train_loss(df=irun_df,
                        arr_list=['gen_loss', 'disc_loss'],
                        figname='%s/training_model_losses.png'%(irun_dir)
                       )


        # Marker attributes
        alpha = 0.65
        mkrsize = 7
        
        fine_epochs = np.linspace(0, n_epochs, 1000)

        # Plot results of xcorr width fits 
        figname='%s/training_sigmaxcorr.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        ax = fig.add_subplot(111)
        epochs = range(0, n_epochs)
        ax.plot(epochs, gen_fit_sigma, color='b', marker='o', alpha=alpha, ms=mkrsize, mec='None', lw=0)
        ax.plot(fine_epochs, np.ones(len(fine_epochs)), 'k--', linewidth=1, alpha=0.5)
        ax.set_ylabel(r'$\sigma^{g}_{\rho} / \sigma^{t}_{\rho}$', fontsize=16)
        ax.set_xlabel(r'Epoch')
        ax.set_title(r'Cross Corr. Width Ratio for $N=%i$ samples'%n_test_samples)
        fig.tight_layout()
        print(figname)
        fig.savefig(figname)


        # Plot results of train/test-gen xcorr width fits 
        figname='%s/training_sigmaxcorr_testtrain.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        ax = fig.add_subplot(111)
        epochs = range(0, n_epochs)
        # Gen-Gen xcorr
        ax.plot(epochs, gen_fit_sigma, color='b', marker='o',
                alpha=alpha-0.15, ms=mkrsize, mec='None', lw=0, label='Gen-Gen')
        # Gen-Train
        ax.plot(epochs, gr_fit_sigma, color='r', marker='s',
                alpha=alpha, ms=mkrsize, mec='None', lw=0, label='Gen-Train')
        # Gen-Test
        ax.plot(epochs, gt_fit_sigma, color='r', marker='s',
                ms=mkrsize, mfc='None', lw=0, label='Gen-Test')
        ax.plot(fine_epochs, np.ones(len(fine_epochs)), 'k--', linewidth=1, alpha=0.5)
        ax.set_ylabel(r'$\sigma^{g}_{\rho} / \sigma^{t}_{\rho}$', fontsize=16)
        ax.set_xlabel(r'Epoch')
        ax.set_title(r'Cross Corr. Width Ratio for $N=%i$ samples'%n_test_samples)
        plt.legend(loc='upper right')
        fig.tight_layout()
        print(figname)
        fig.savefig(figname)


        # Plot results of KS compared to train data
        figname='%s/training_kstrain.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        axd = fig.add_subplot(111)
        epochs = range(0, n_epochs)
        # D-Values
        axd.plot(epochs, dvalues_train, label=r'$KS_{D}$', color='b',
                 alpha=alpha, marker='o', mec='None', ms=mkrsize, lw=0)
        axd.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$', color='b')
        axd.tick_params('y', colors='b')
        axd.set_xlabel(r'Epoch')
        axd.set_title(r'KS on Euclidean Norm - Train Data')
        # P-Values
        axp = axd.twinx()
        axp.plot(epochs, pvalues_train, label=r'$KS_{p}$', color='r',
                 alpha=alpha, marker='s', mec='None', ms=mkrsize, lw=0)
        axp.set_ylim(0, 1.1*max(pvalues_train))
        axp.set_ylabel(r'$\mathrm{KS}_{\mathrm{p}}$', color='r')
        axp.tick_params('y', colors='r')
        fig.tight_layout()
        print(figname)
        fig.savefig(figname)

        
        # Plot results of KS compared to test data
        figname='%s/training_kstest.png'%(irun_dir)
        fig = plt.figure(figsize=(9,6))
        mpl.rc("font", family="serif")
        axd = fig.add_subplot(111)
        epochs = range(0, n_epochs)
        # D-Values
        axd.plot(epochs, dvalues_test, label=r'$KS_{D}$', color='b',
                 alpha=alpha, marker='o', mec='None', ms=mkrsize, lw=0)
        axd.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$', color='b')
        axd.tick_params('y', colors='b')
        axd.set_xlabel(r'Epoch')
        axd.set_title(r'KS on Euclidean Norm - Test Data')
        # P-Values
        axp = axd.twinx()
        axp.plot(epochs, pvalues_test, label=r'$KS_{p}$', color='r',
                 alpha=alpha, marker='s', mec='None', ms=mkrsize, lw=0)
        axp.set_ylim(0, 1.1*max(pvalues_test))
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
        axd.plot(epochs, dvalues_test, label=r'$\mathrm{KS}_{D}^{\mathrm{Test}}$', color='b',
                 marker='o', alpha=alpha, mec='None', ms=mkrsize, lw=0)
        axd.plot(epochs, dvalues_train, label=r'$\mathrm{KS}_{D}^{\mathrm{Train}}$', color='b',
                 marker='o', mfc='None', ms=mkrsize, lw=0)
        axd.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$', color='b')
        axd.tick_params('y', colors='b')
        axd.set_xlabel(r'Epoch')
        title_axd = '$\mathrm{KS}_{\mathrm{D}}$, $\sigma^{g,t}_{\\rho}$ Ratio $(n=%i,\ d=%i,\ f=%i)$'%(irun_df['dim'],
                                                                                                      irun_df['latent_dim'],
                                                                                                      irun_df['scale_factor'])
        axd.set_title(r'%s'%title_axd)
        plt.legend(loc='upper right')
        # Fit widths of xcorr on generated samples
        axp = axd.twinx()
        axp.plot(epochs, gen_fit_sigma, color='r', marker='s', alpha=alpha, ms=mkrsize, lw=0, mec='None')
        axp.set_ylabel(r'$\sigma^{g}_{\rho} / \sigma^{t}_{\rho}$', fontsize=16, color='r')
        axp.tick_params('y', colors='r')
        fig.tight_layout()
        print(figname)
        fig.savefig(figname)
    

if __name__ == "__main__":
    main()
