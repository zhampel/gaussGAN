from __future__ import print_function

try:
    import os
    import numpy as np

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
except ImportError as e:
    print(e)
    raise ImportError


def compare_histograms(hist_list=[], centers=[], labels=[], ylims=[0, 1, False], figname=None):

    fig = plt.figure(figsize=(9,6))
    mpl.rc("font", family="serif")
    ax = fig.add_subplot(111)
    ymax = -1.0
    
    for idx, hist in enumerate(hist_list):
        # Draw dist with steps
        ax.step(centers[idx], hist, linewidth=1.5, label=labels[idx])
        ymax = max(ymax, np.float(np.max(hist)))

    ax.set_xlabel(r'Radius, $r = |x|_{2}$')
    ax.set_ylim(ylims[0], ylims[1])
    
    ax.set(xlabel=r'Vector Magnitude, $r$', ylabel='P(r)',
           title='Distribution of Vector Magnitudes')

    ax.grid()

    if ylims[2] and ylims[0] > 0.0:
        plt.yscale('log')

    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    fig.savefig(figname)


def plot_histogram(hist=None, centers=None, label=None, figname=None):

    fig = plt.figure(figsize=(9,6))
    mpl.rc("font", family="serif")
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
        ax.plot(epochs, vals, label=r'%s'%(label))
    
    ax.set(xlabel='Epoch', ylabel='Loss',
           title='Loss vs Training Epoch')
    ax.grid()
    #plt.yscale('log')
    plt.legend(loc='upper right', fontsize=16)
    print(figname)
    plt.tight_layout()
    fig.savefig(figname)
