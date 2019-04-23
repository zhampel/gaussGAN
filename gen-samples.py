from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    from scipy.optimize import curve_fit

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import ScalarFormatter

    import pandas as pd
    
    import torch
    
    from gaussgan.definitions import RUNS_DIR
    from gaussgan.utils import sample_z
except ImportError as e:
    print(e)
    raise ImportError

    
# Marker/Line Options
colors = ["blue", "red", "green", "black"]
colorsmall = ["b", "r", "g", "k"]
styles = ["-", "--", "-."]


def gaussian(x, mu, sig):
    """
    Gaussian function
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-b", "--n_batches", dest="n_batches", default=100, type=int, help="Number of batches")
    parser.add_argument("-n", "--n_samples", dest="n_samples", default=1024, type=int, help="Number of samples")
    args = parser.parse_args()

    # Number of samples to take
    n_samples = args.n_samples
    n_batches = args.n_batches
    n_total = int(n_samples * n_batches)

    # List of dimensions to test
    dim_list = [1, 2, 3, 10, 50, 100, 1000]
    n_dims = len(dim_list) 

    # Define histograms for saving generated data
    redges = np.arange(0, 50, 0.25)
    rcents = (redges[1:]-redges[:-1])/2 + redges[0:-1]
    inv_w = 1 / float(n_total)
   
    # Lists for saving histograms
    rhist_list = []

    # Loop through dimensions to run
    for idx, dim in enumerate(dim_list):

        # Initialize histogram for dimensionality
        rhist, _ = np.histogram([], bins=redges)

        # Run through number of batches, getting n_samples each
        for ibatch in range(n_batches):
            z = sample_z(samples=n_samples, dims=dim, mu=0.0, sigma=1.0)
            r2 = torch.sum(z * z, dim=1)
            r = torch.sqrt(r2)
            # Add to histogram
            rhist += np.histogram(r.cpu().data.numpy(), bins=redges)[0]

        # Save histogram to list
        rhist_list.append(rhist)


    # Euclidean distance distributions
    ylab = '$P(r)$'
    title = 'Radius Distributions of Multi-$d$ Gaussian $x \in \mathcal{R}^d$'

    fig = plt.figure(figsize=(9,6))
    mpl.rc("font", family="serif")
    ax = fig.add_subplot(111)
    ax.set_xlim(redges[0], redges[-1])
    ymax = -1.0

    for idx, dim in enumerate(dim_list):
        rlab = '$%i$'%dim
        # Normalize histogram
        rhist = rhist_list[idx] / float(n_total)
        # Draw dist with steps
        ax.step(rcents, rhist, linewidth=1.5, label=rlab)
        ymax = max(ymax, np.float(np.max(rhist)))

    ax.set_xlabel(r'Radius, $r = |x|_{2}$')
    #ax.set_ylim(1/float(n_total), 1.1*ymax)
    ax.set_ylim(0, 1.1*ymax)
    ax.set_ylabel(r'%s'%ylab)
    ax.set_title(r'%s'%title)
    plt.legend(title=r'Dimension, $d$', loc='best', numpoints=1, ncol=2)
    #ax.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    fig.savefig('g_distribution.png')


    # Gaussian fits to distributions
    rmean_fit = np.zeros((len(dim_list)))
    rmean_fit_err = np.zeros((len(dim_list)))
    rsigma_fit = np.zeros((len(dim_list)))
    rsigma_fit_err = np.zeros((len(dim_list)))

    for idx, dim in enumerate(dim_list):
        # Analytic mean r
        analytic_mean = np.sqrt(dim)
        # Normalixze histogram
        rhist = rhist_list[idx] / float(n_total)
        #rhist_unc = np.sqrt(rhist_list[idx]) / float(n_total)
        # Perform gaussian fit
        popt, pcov = curve_fit(gaussian, rcents, rhist, bounds=(0, [2*analytic_mean, analytic_mean]))
        # Calculate uncertainties from fit
        perr = np.sqrt(np.diag(pcov))
        # Save results in arrays
        rmean_fit[idx] = popt[0]
        rmean_fit_err[idx] = perr[0]
        rsigma_fit[idx] = popt[1]
        rsigma_fit_err[idx] = perr[1]
        print("Dim: %i\n\tmean r: %.03f ± %.03f\tsigma r: %.03f ± %.03f"%(dim, popt[0], perr[0], popt[1], perr[1]))

    # Calculate analytic mean for continuous curve
    lin_dims = np.arange(1, np.max(dim_list), 1)
    rmean_ana = np.sqrt(lin_dims)

    # Plot results
    title = 'Mean Radius of Multi-d Gaussian $x \in \mathcal{R}^d$'
    fig = plt.figure(figsize=(9,6))
    mpl.rc("font", family="serif")
    ax1 = fig.add_subplot(111)
    # Mean distance
    ax1.plot(lin_dims, rmean_ana, label=r'$\sqrt{d}$')
    ax1.errorbar(dim_list, rmean_fit, yerr=rmean_fit_err, label=r'Fit', fmt='o', markersize=8)
    ax1.set_xlabel(r'Dimensionality, $d$')
    ax1.set_ylim(0, 1.5*rmean_ana[-1])
    ax1.set_xlim(0.9, 1.1*lin_dims[-1])
    ax1.set_ylabel(r'Mean Radius $\overline{|x|}_{2}$')
    ax1.set_title(r'%s'%title)
    plt.legend(title=r'$\overline{|x|}$', loc='upper left', numpoints=1)
    ax1.set_xscale('log', nonposx='clip')
    # Sigma of distance distribution
    ax2 = ax1.twinx()
    ax2.errorbar(dim_list, rsigma_fit, yerr=rsigma_fit_err, label='Fit', fmt='o', markersize=8, color='r')
    ax2.set_ylim(0, 1.3*np.max(rsigma_fit))
    ax2.set_ylabel(r'Width of Radius Distribution $\sigma_{|x|}$', color='r')
    ax2.tick_params('y', colors='r')
    plt.legend(title=r'$\sigma_{|x|}$', loc='upper right', numpoints=1)
    fig.tight_layout()
    fig.savefig('g_rmeans.png')


    # Save fit results
    train_df = pd.DataFrame({
                             'dims' : dim_list,
                             'n_samples' : n_total,
                             'rhist_list' : rhist_list,
                             'rmean_fit' : rmean_fit,
                             'rmean_fit_err' : rmean_fit_err,
                             'rsigma_fit' : rsigma_fit,
                             'rsigma_fit_err' : rsigma_fit_err,
                            })

    train_df.to_csv('fit_details.csv')


if __name__ == "__main__":
    main()
