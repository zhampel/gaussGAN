from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm as cm

    import pandas as pd

    import re as re
    
except ImportError as e:
    print(e)
    raise ImportError


def scatter_plot(x=[], y=[], c=[], title='', xlabel='', ylabel='', clabel='', figname=''):
    """
    Function to save a scatter plot with colored markers respective of z value
    """
    cmap = cm.get_cmap('Reds', 50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=c, marker='o', cmap=cmap, lw=0, norm=matplotlib.colors.LogNorm())
    ax.set_title(r'%s'%title, fontsize=16)
    ax.set_xlabel(r'%s'%xlabel, fontsize=16)
    ax.set_ylabel(r'%s'%ylabel, fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    cbar = plt.colorbar(sc)
    cbar.set_label(r'%s'%clabel, fontsize=16)
    fig.tight_layout()
    fig.savefig(figname)


def main():
    global args
    parser = argparse.ArgumentParser(description="GAN metric plotting script")
    parser.add_argument("-r", "--run_list", dest="run_list", nargs="+", help="List of runs")
    parser.add_argument("-s", "--suffix", dest="suffix", default="", help="Suffix to figure file names")
    args = parser.parse_args()

    # Get directories as run list
    run_list = args.run_list
    # Get suffix to names of output files
    suffix = args.suffix
    if suffix is not "":
        suffix = "_" + suffix

    # Declare some lists
    latent_list = []
    dscale_list = []
    sigma_list = []
    dvalue_list = []
    sigma_best_list = []
    dvalue_best_list = []
    sum_best_list = []

    # Theoretical sigma from true samples
    n_samp = 1000
    sigma_theo = 1/np.sqrt(n_samp)

    # Pass through run directories
    for irun_dir in run_list:
   
        # Currently needed to extract scale parameter
        int_vals = re.findall('\d+', irun_dir)
        int_vals = [int(i) for i in int_vals]

        train_deets = '%s/training_details.csv'%(irun_dir)

        # Skip if training file not found
        if not os.path.isfile(train_deets):
            print("\n\t%s DNE. Skipping...\n"%train_deets)
            continue

        # Read in training information from run
        irun_df = pd.read_csv(train_deets)
        # Get data, latent dims and scale value
        dim = irun_df['dim'][0]
        latent_dim = irun_df['latent_dim'][0]
        # Grab scale value from name.... will put in training file next update
        dscale = int_vals[1]
        # Get KS distance values
        dvalues_str = irun_df['dvalues'][1]
        dvalues_list = dvalues_str[1:-1].split(',')
        dvalues = [float(val) for val in dvalues_list]
        # Get Generated sigma fits
        gen_fit_sigma_str = irun_df['gen_fit_sigma'][1]
        gen_fit_sigma_list = gen_fit_sigma_str[1:-1].split(',')
        # Normalize by theoretical sigma from true samples
        gen_fit_sigma = [float(val)/sigma_theo for val in gen_fit_sigma_list]

        # Save latent and scale values
        latent_list.append(float(latent_dim))
        dscale_list.append(float(dscale))

        #dvalues /= np.max(dvalues)
        #gen_fit_sigma /= np.max(gen_fit_sigma)

        # Save final sigma and KS_D
        sigma_list.append(gen_fit_sigma[-1])
        dvalue_list.append(dvalues[-1])
    
        # Combine normalized sigmas and KSDs to find global optimum of run
        sum_array = (gen_fit_sigma/np.max(gen_fit_sigma) + dvalues/np.max(dvalues))/2
        idxmin = np.argmin(sum_array)

        # Save joint optimal sigma and KS_D, joint value
        sigma_best_list.append(gen_fit_sigma[idxmin])
        dvalue_best_list.append(dvalues[idxmin])
        sum_best_list.append(sum_array[idxmin])


    sclabel = 'Scale Factor'
    ldlabel = 'Latent Dim'

    # Save scattered values to figures

    # Final epoch sigma
    scatter_plot(x=dscale_list, y=latent_list, c=sigma_list,
                 title='Final $\sigma_{\\rho}$ Value',
                 xlabel=sclabel, 
                 ylabel=ldlabel, 
                 clabel='$\sigma^{g}_{\\rho} / \sigma^{t}_{\\rho}$', 
                 figname='dim50_final_sigma%s.png'%suffix)

    # Final epoch KSd
    scatter_plot(x=dscale_list, y=latent_list, c=dvalue_list,
                 title='Final KS$_D$ Value',
                 xlabel=sclabel, 
                 ylabel=ldlabel, 
                 clabel='KS$_D$', 
                 figname='dim50_final_dvalue%s.png'%suffix)
   
    # Best sigma
    scatter_plot(x=dscale_list, y=latent_list, c=sigma_best_list,
                 title='Best $\sigma_{\\rho}$ Value',
                 xlabel=sclabel, 
                 ylabel=ldlabel, 
                 clabel='$\sigma^{g}_{\\rho} / \sigma^{t}_{\\rho}$', 
                 figname='dim50_best_sigma%s.png'%suffix)

    # Best KSd
    scatter_plot(x=dscale_list, y=latent_list, c=dvalue_best_list,
                 title='Best KS$_D$ Value',
                 xlabel=sclabel, 
                 ylabel=ldlabel, 
                 clabel='KS$_D$', 
                 figname='dim50_best_dvalue%s.png'%suffix)

    # Global optimum
    scatter_plot(x=dscale_list, y=latent_list, c=sum_best_list,
                 title='Min KS$_D + \sigma_{\\rho}$ per Run',
                 xlabel=sclabel, 
                 ylabel=ldlabel, 
                 clabel='$\sum$ Max-Scaled Metrics', 
                 figname='dim50_global_optimum%s.png'%suffix)

    # Identify global joint optimum from ALL runs
    idxmin = np.argmin(sum_best_list)
    print("Global Optimum:")
    print("Scale Factor: %.1f"%dscale_list[idxmin])
    print("Latent Dim: %i"%latent_list[idxmin])
    print("Sigma Ratio: %.04f"%sigma_best_list[idxmin])
    print("KS Dist: %.04f"%dvalue_best_list[idxmin])
    

if __name__ == "__main__":
    main()
