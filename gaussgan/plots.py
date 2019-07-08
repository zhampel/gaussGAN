from __future__ import print_function

try:
    import os
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.stats import norm

    import matplotlib
    import matplotlib as mpl
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    from matplotlib import cm as cm

    from gaussgan.utils import gaussian, smooth
    
except ImportError as e:
    print(e)
    raise ImportError

colors = ['k', 'r', 'b', 'm']
ncols = len(colors)
mpl.rc("font", family="serif", size=14)

def plot_corr(corr=None, figname='', title='', comp_hist=None):

    fig = plt.figure(figsize=(18,6))
    
    ax = fig.add_subplot(121)
    cmap = cm.get_cmap('coolwarm', 30)
    cp = ax.pcolormesh(corr, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_title(r'%s'%title)
    ax.set_xlabel(r'Component $i$')
    ax.set_ylabel(r'Component $j$')
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label(r'$\rho_{ij}$', fontsize=16)
   
    # Plot histogram of off-diag corr
    off_diag = np.ravel(corr[~np.eye(corr.shape[0], dtype=bool)])
    xedges = np.linspace(-1.0, 1.0, 100)
    xcents = (xedges[1:]-xedges[:-1])/2 + xedges[0:-1]
    off_hist = np.histogram(off_diag, bins=xedges)[0]
    off_hist = off_hist / np.sum(off_hist)
    off_hist = np.append(off_hist, off_hist[-1])

    ax = fig.add_subplot(122)
    if comp_hist is not None:
        ax.step(xedges, off_hist, c='r', where='post', label=r'$\rho^{g}$')
        ax.step(xedges, comp_hist, 'k--', where='post', label=r'$\rho^{t}$')

    else:
        ax.step(xedges, off_hist, c='k', where='post', label=r'$\rho^{t}$')

    # Plot fit to normal distribution
    (mu, sigma) = norm.fit(off_diag)
    xfit = np.linspace(-1.0, 1.0, 1000)
    gfit = np.max(off_hist) * gaussian(xfit, mu, sigma)
    ax.plot(xfit, gfit, 'b:', linewidth=2.0, label=r'$\mathscr{N}(%.03f, %.03f)$'%(mu, sigma))

    ax.set_xlabel(r'$\rho(x_i, x_j)$, $i \neq j$')
    ax.set_ylabel(r'Normalized Frequency')
    ax.set_xlim(-1.0, 1.0)
    plt.yscale('log')
    ax.set_ylim(0.001, 1.0)
    #ax.set_ylim(0.0, 0.5)
    ax.grid()
    plt.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(figname)
    return off_hist, sigma


def compare_histograms(hist_list=[], centers=[], labels=[], ylims=[0, 1, False], figname=None):

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ymax = -1.0
    
    for idx, hist in enumerate(hist_list):
        # Draw dist with steps
        ax.step(centers[idx], hist, linewidth=1.5, label=labels[idx], c=colors[idx%ncols])
        ymax = max(ymax, np.float(np.max(hist)))

    ax.set_xlabel(r'$\mathscr{l}^{2}$-Norm, $r = ||\mathbf{x}||_{2}$')
    ax.set_ylabel(r'Normalized Frequency')
    ax.set_ylim(ylims[0], ylims[1])
    ax.set_title('Magnitude Distributions at Epoch %s'%labels[-1])
    
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

    # Get data, latent dims and scale value
    dim = df['dim']
    n_epochs = df['n_epochs']
    latent_dim = df['latent_dim']
    dscale = df['scale_factor']

    fig, ax = plt.subplots(figsize=(16,10))
    for arr in arr_list:
        label = df[arr][0]
        vals = df[arr][1]
        n_iter = len(vals)
        epochs = np.linspace(0., np.float(n_epochs), num=n_iter)
        ax.plot(epochs, vals, label=r'$%s$'%label, marker='o', markersize=3, linewidth=0)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(r'Training Losses $(n=%i,\ d=%i,\ f=%i)$'%(dim, latent_dim, dscale))
    #ax.set_title(r'Training Losses $(n=%i,\ d=%i,\ f=%i)$'%(df['dim'], df['latent_dim'], df['scale_factor']))
    ax.grid()
    plt.legend(loc='upper right', fontsize=16)
    print(figname)
    plt.tight_layout()
    fig.savefig(figname)


def plot_train_curves(df=[], arr_list=[''], figname='training_curves.png'):

    n_epochs = df['n_epochs']
    loss_list = [v for v in arr_list if 'loss' in v]
    mean_list = [v for v in arr_list if 'mean' in v]

    cidx = 0
    fig, axl = plt.subplots(figsize=(16,10))
    for arr in loss_list:
        label = df[arr][0]
        vals = df[arr][1]
        n_iter = len(vals)
        epochs = np.linspace(0., np.float(n_epochs), num=n_iter)
        axl.plot(epochs, vals, label=r'$%s$'%(label), marker='o', markersize=1, linewidth=0, alpha=0.25, color=colors[cidx])
        #smoothed = smooth(x=np.asarray(vals), window_len=10, window='flat')
        smoothed = smooth(x=np.asarray(vals), window_len=2*df['batch_size'], window='flat')
        axl.plot(np.linspace(0., np.float(n_epochs), num=len(smoothed)), smoothed, '--', linewidth=1, color=colors[cidx])
        cidx += 1

    axl.set_ylabel('Loss')
    axl.set_xlabel('Epoch')
    axl.set_title(r'Training Curves $(n=%i,\ d=%i,\ f=%i)$'%(df['dim'], df['latent_dim'], df['scale_factor']))
    axl.grid()
    axl.legend(loc='upper left', title='Loss', fontsize=16)

    axm = axl.twinx()
    for arr in mean_list:
        label = df[arr][0]
        vals = df[arr][1]
        n_iter = len(vals)
        epochs = np.linspace(0., np.float(n_epochs), num=n_iter)
        axm.plot(epochs, vals, label=r'$%s$'%(label), marker='s', markersize=2, linewidth=0, alpha=0.25, color=colors[cidx])
        #smoothed = smooth(x=np.asarray(vals), window_len=10, window='flat')
        smoothed = smooth(x=np.asarray(vals), window_len=2*df['batch_size'], window='flat')
        axm.plot(np.linspace(0., np.float(n_epochs), num=len(smoothed)), smoothed, '-', linewidth=1, color=colors[cidx])
        cidx += 1

    axm.set_ylim(0, 1)
    axm.set_ylabel(r'$\overline{D}(x)$')
    axm.legend(loc='upper right', title='Disc. Out', fontsize=16)
    
    #plt.legend(loc='upper right', fontsize=16)
    print(figname)
    plt.tight_layout()
    fig.savefig(figname)



def plot_component_ks(figname='ks_epoch00000.png', epoch=0, dim=1, ks_d_list=[], ks_p_list=[]):

    fig = plt.figure(figsize=(9,6))
    mpl.rc("font", family="serif")
    axd = fig.add_subplot(111)
    # D-Values
    axd.step(np.arange(-0.5, dim+0.5, 1), ks_d_list, c='k', where='post')
    axd.set_ylabel(r'$\mathrm{KS}_{\mathrm{D}}$')
    axd.set_xlabel(r'Vector Component')
    axd.set_title(r'KS Test for Each Component at Epoch %i'%epoch)
    # P-Values
    axp = axd.twinx()
    axp.step(np.arange(-0.5, dim+0.5, 1), ks_p_list, c='r', where='post')
    axp.set_ylabel(r'$\mathrm{KS}_{\mathrm{p}}$', color='r')
    axp.tick_params('y', colors='r')
    fig.tight_layout()
    fig.savefig(figname)

