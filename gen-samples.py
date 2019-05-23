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

    import tables

    import pandas as pd
    
    import torch
    
    from gaussgan.definitions import DATASETS_DIR
    from gaussgan.utils import gaussian, Sampler, enorm, sampler_list
    from gaussgan.plots import plot_histogram, compare_histograms
except ImportError as e:
    print(e)
    raise ImportError


# Marker/Line Options
colors = ["blue", "red", "green", "black"]
colorsmall = ["b", "r", "g", "k"]
styles = ["-", "--", "-."]


def main():
    global args
    parser = argparse.ArgumentParser(description="Dataset generation script")
    parser.add_argument("-b", "--n_batches", dest="n_batches", default=100, type=int, help="Number of batches")
    parser.add_argument("-n", "--n_samples", dest="n_samples", default=1024, type=int, help="Number of samples")
    parser.add_argument("-d", "--dim_list", dest="dim_list", nargs='+', default=[1, 2, 3, 10, 100], type=int, help="Number of samples")
    parser.add_argument("-s", "--dist_name", dest="dist_name", default='gauss', choices=sampler_list,  help="Sampling distribution name")
    args = parser.parse_args()

    # Make directory structure for this run
    data_dir = os.path.join(DATASETS_DIR)
    os.makedirs(data_dir, exist_ok=True)
    print('\nDatasets to be saved in directory %s\n'%(data_dir))
    atom = tables.Float64Atom()

    # Distribution name
    dist_name = args.dist_name

    # Distribution parameters
    mu = 0.0
    sigma = 1.0
    xlo = -1.5
    xhi = 1.5

    # Number of samples to take
    n_samples = args.n_samples
    n_batches = args.n_batches
    n_total = int(n_samples * n_batches)
    
    # List of dimensions to test
    dim_list = args.dim_list
    n_dims = len(dim_list)

    print("Generating data for dimensions: ", dim_list)
    print("Each dataset will have %i samples."%n_total)

    # Define histograms for saving generated data
    redges = np.arange(0, 50, 0.25)
    rcents = (redges[1:]-redges[:-1])/2 + redges[0:-1]
   
    # Lists for saving histograms
    rhist_list = []

    # Loop through dimensions list
    for idx, dim in enumerate(dim_list):

        # Initialize histogram for dimensionality
        rhist, _ = np.histogram([], bins=redges)
        
        sampler = Sampler(dist_name=dist_name, dim=dim, n_samples=n_samples, mu=mu, sigma=sigma, xlo=xlo, xhi=xhi)

        # Prepare file, earray to save generated data
        data_file_name = '%s/data_%s_dim%i.h5'%(data_dir, dist_name, dim)
        data_file = tables.open_file(data_file_name, mode='w')

        meta_group = data_file.create_group(data_file.root, 'metadata')
        data_file.create_array(meta_group, 'dim', np.asarray([dim]), 'Dimensionality')
        data_file.create_array(meta_group, 'n_batches', np.asarray([n_batches]), 'Number of Batches')
        data_file.create_array(meta_group, 'n_samples', np.asarray([n_samples]), 'Number of Samples per Batch')
        data_file.create_array(meta_group, 'n_total', np.asarray([n_total]), 'Total Number of Samples')
        data_file.create_array(meta_group, 'mu', np.asarray([mu]), 'Mean')
        data_file.create_array(meta_group, 'sigma', np.asarray([sigma]), 'Sigma')

        array_c = data_file.create_earray(data_file.root, 'data', atom, (0, dim))

        # Run through number of batches, getting n_samples each
        for ibatch in range(n_batches):
            # Random set of n_samples
            z = sampler.sample()
            z_numpy = z.cpu().data.numpy()
            # Calculate magnitude of vector, follows chi dist
            r_numpy = enorm(z)

            # Add norm entry to histogram
            rhist += np.histogram(r_numpy, bins=redges)[0]
            # Add samples dataset
            array_c.append(z_numpy)
 
        # Save histogram to list
        rhist_list.append(rhist)

        # Close dataset file
        data_file.close()


    ## Generate figures
    # Euclidean distance distributions
    norm_histlist = [rhist/float(n_total) for rhist in rhist_list]
    # Normal y-axis
    compare_histograms(hist_list=norm_histlist,
                       centers=[rcents]*len(rhist_list),
                       labels=dim_list,
                       ylims=[0.0, 0.2, False],
                       figname='g_%s_distribution.png'%dist_name)
    
    # Logarithmic y-axis
    compare_histograms(hist_list=norm_histlist,
                       centers=[rcents]*len(rhist_list),
                       labels=dim_list,
                       ylims=[0.0001, 0.2, True],
                       figname='g_%s_distribution_log.png'%dist_name)

    # Plot each distribution separately
    for idx, dim in enumerate(dim_list):
        figname = '%s/hist_%s_dim%i.png'%(data_dir, dist_name, dim)
        plot_histogram(hist=rhist_list[idx]/float(n_total),
                       centers=rcents,
                       label=r'$d=%i$'%dim,
                       figname=figname)

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
    ax1.errorbar(dim_list, rmean_fit, yerr=rmean_fit_err, label=r'Fit', fmt='s', markersize=8)
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
    fig.savefig('g_%s_rmeans.png'%dist_name)


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

    train_df.to_csv('data_%s_details.csv'%dist_name)


if __name__ == "__main__":
    main()
