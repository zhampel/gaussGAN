from __future__ import print_function

try:
    import os
    import numpy as np
    from scipy.optimize import curve_fit

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm as cm

    from gaussgan.utils import gaussian
    
except ImportError as e:
    print(e)
    raise ImportError

colors = ['k', 'r', 'b', 'm']
mpl.rc("font", family="serif", size=14)

def plot_corr(corr=None, figname='', title='', comp_hist=None):

    fig = plt.figure(figsize=(18,6))
    
    ax = fig.add_subplot(121)
    cmap = cm.get_cmap('coolwarm', 30)
    cp = ax.pcolormesh(corr, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_title(r'%s'%title)
    ax.set_xlabel(r'Component')
    ax.set_ylabel(r'Component')
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label(r'$r(x_i, x_j)^{g}$', fontsize=16)
   
    # Plot histogram of off-diag corr
    off_diag = np.ravel(corr[~np.eye(corr.shape[0], dtype=bool)])
    xedges = np.linspace(-1.0, 1.0, 100)
    xcents = (xedges[1:]-xedges[:-1])/2 + xedges[0:-1]
    off_hist = np.histogram(off_diag, bins=xedges)[0]
    off_hist = off_hist / np.sum(off_hist)
    off_hist = np.append(off_hist, off_hist[-1])

    ## Perform gaussian fit
    #popt, pcov = curve_fit(gaussian, xcents, off_hist[0:-1], bounds=([-0.5, 0.00001], [0.5, 0.5]))
    ## Calculate uncertainties from fit
    #perr = np.sqrt(np.diag(pcov))
    #mu = popt[0]
    #sigma = popt[1]
    #xfit = np.linspace(-1.0, 1.0, 1000)
    #gfit = gaussian(xfit, mu, sigma)
    #ax.plot(xfit, gfit, color='r', linewidth=0.75, label=r'Fit: $N(%f, %f)$'%(mu, sigma))

    ax = fig.add_subplot(122)
    if comp_hist is not None:
        ax.step(xedges, off_hist, c='r', where='post', label=r'$r_{x,y}^{g}$')
        ax.step(xedges, comp_hist, 'k--', where='post', label=r'$r_{x,y}^{t}$')
    else:
        ax.step(xedges, off_hist, c='k', where='post', label=r'$r_{x,y}$')

    ax.set_xlabel(r'$r(x_i, x_j)$, $i \neq j$')
    ax.set_ylabel(r'Normalized Frequency')
    ax.set_xlim(-1.0, 1.0)
    plt.yscale('log')
    ax.set_ylim(0.005, 1.0)
    ax.grid()
    plt.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(figname)
    return off_hist


def compare_histograms(hist_list=[], centers=[], labels=[], ylims=[0, 1, False], figname=None):

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ymax = -1.0
    
    for idx, hist in enumerate(hist_list):
        # Draw dist with steps
        ax.step(centers[idx], hist, linewidth=1.5, label=labels[idx], c=colors[idx])
        ymax = max(ymax, np.float(np.max(hist)))

    ax.set_xlabel(r'$\mathscr{l}^{2}$-Norm, $r = ||\mathbf{x}||_{2}$')
    ax.set_ylabel(r'Normalized Frequency')
    ax.set_ylim(ylims[0], ylims[1])
    ax.set_title('Distribution of Vector Magnitudes')
    
    ax.grid()

    if ylims[2] and ylims[0] > 0.0:
        plt.yscale('log')

    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    fig.savefig(figname)


def plot_histogram(hist=None, centers=None, label=None, figname=None):

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ymax = -1.0
    # Draw dist with steps
    ax.step(centers, hist, linewidth=1.5, label=label)
    ymax = max(ymax, np.float(np.max(hist)))

    ax.set_xlabel(r'Radius, $r = |x|_{2}$')
    ax.set_ylim(0, 1.2*ymax)
    
    ax.set(xlabel=r'Vector Magnitude, $r$', ylabel='P(r)',
           title='Distribution of Vector Magnitudes')

    ax.grid()
    #plt.yscale('log')
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    fig.savefig(figname)


def plot_train_loss(df=[], arr_list=[''], figname='training_loss.png'):

    fig, ax = plt.subplots(figsize=(16,10))
    for arr in arr_list:
        label = df[arr][0]
        vals = df[arr][1]
        epochs = range(0, len(vals))
        ax.plot(epochs, vals, label=r'%s'%(label), marker='o', linewidth=0)
    
    ax.set(xlabel='Epoch', ylabel='Loss',
           title='Loss vs Training Epoch')
    ax.grid()
    #plt.yscale('log')
    plt.legend(loc='upper right', fontsize=16)
    print(figname)
    plt.tight_layout()
    fig.savefig(figname)
