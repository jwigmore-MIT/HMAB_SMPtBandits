import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from scipy.stats import norm




def plot_rv(rv,fig = None, ax = None, label=None, color = None):
    if ax is None:
        fig, ax = plt.subplots()

    a = .005  # Quantiles for plotting
    points = 1000  # Number of points to plot
    x = np.linspace(rv.ppf(a), rv.ppf(1 - a), points)
    if color is not None:
        p = ax.plot(x, rv.pdf(x), label=label, color = color)
    else:
        p = ax.plot(x, rv.pdf(x), label=label)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    return (fig,ax)

def plot_bins(bins, fig = None, ax = None):
    plt.rcParams['figure.figsize'] = [12, 5]

    if ax is None:
        fig, ax = plt.subplots(1,2)
    stem_height = 0.1
    ax[0].stem(bins, np.ones(bins.shape[0])*stem_height)
    ax[0].set_title("Expected Bin Rewards $\mathbf{\mu}_{i,j}$")
    ax[0].set_ylim(0,0.5)
    ax[1].hist(bins, density = True, )
    ax[1].set_title("Histogram of Expected Bin Rewards")

    return (fig,ax)

def plot_env(Band, Bins, fig = None, ax= None):
    plt.rcParams['figure.figsize'] = [16, 5]

    if ax is None:
        fig, ax = plt.subplots(1,3)
    plot_rv(Band, ax = ax[0])
    (mu_i,sigma_i) = Band.stats()
    ax[0].set_title(f"Generative Distribution for Band $B_i \sim N(\mu_i = {mu_i}, \sigma_i^2 = {sigma_i**2}$)")
    stem_height = 0.1
    ax[1].stem(Bins, np.ones(Bins.shape[0]) * stem_height)
    ax[1].set_title("Expected Bin Rewards $\mathbf{\mu}_{i,j}$")
    ax[1].set_ylim(0, 0.5)
    ax[2].hist(Bins, density=True,alpha = 0.5)
    emp_rv = norm(np.mean(Bins), 1)
    plot_rv(emp_rv,ax = ax[2])
    ax[2].set_title("Histogram of Expected Bin Rewards")

def moving_avg(mu_vec, n):
    #n = mu_vec.shape[1]
    if n ==0:
        roll_avg =  mu_vec[0]
    else:
        roll_avg =  (np.convolve(mu_vec,np.ones(n+1), 'valid')/(n+1))[0]
    return roll_avg


def plot_est_bins(bins, ests, E1, E2, maxK = None):
    #plt.rcParams['figure.figsize'] = [18, 12]
    if maxK is None:
        maxK = bins.shape[0]
    fig, ax = plt.subplots(2,3)
    stem_height = 0.1
    ax[0,0].stem(bins, np.ones(bins.shape[0])*stem_height)
    ax[0,0].set_title("Expected Bin Rewards $\mathbf{\mu}_{i,j}$")
    ax[0,0].set_ylim(0,0.5)

    # Estimated \mu_i
    mu_i = np.mean(bins)
    band_rv = norm(loc=mu_i, scale=1)
    plot_rv(band_rv, ax=ax[0, 1])
    ax[0, 1].hist(bins, density=True)
    ax[0, 1].set_title("Histogram of Expected Bin rewards")
    ax[0, 1].set_xlim(0, 10)

    # For ylims
    minY = np.min([np.min(E1),np.min(E2)]) - 0.1
    maxY =  np.max([np.max(E1), np.max(E2)]) + 0.1

    #E1_mod = np.pad(E1,(maxK-E1.shape[0]), mode = 'empty')
    temp = np.empty(maxK-E1.shape[0])
    temp[:] = np.nan
    #print(temp)
    E1_mod = np.concatenate((E1, temp))
    #print(f"E1_mod: {E1_mod}")
    ax[0,2].scatter(range(maxK),E1_mod)
    ax[0,2].set_ylim(minY, maxY)
    ax[0,2].set_xlim(-1,maxK)
    ax[0,2].set_title(f"E1(K)={round(E1[-1],3)}")


    ax[1,0].stem(ests, np.ones(bins.shape[0])*stem_height)
    ax[1,0].set_title("MLE Estimates of Bin Rewards $\hat{\mathbf{\mu}}_{i,j}$")
    ax[1,0].set_ylim(0,0.5)



    est_mu_i = np.mean(ests)
    est_band_rv = norm(loc=est_mu_i, scale = 1)
    plot_rv(est_band_rv,ax = ax[1,1])
    ax[1,1].hist(ests, density = True )
    ax[1,1].set_title("Histogram of Estimated Expected Bin rewards")
    ax[1,1].set_xlim(0,10)

    E2_mod = np.concatenate((E2, temp))
    ax[1,2].scatter(range(maxK),E2_mod)
    ax[1,2].set_title(f"E2(K) = {round(E2[-1],3)}")
    ax[1,2].set_ylim(minY, maxY)
    ax[1,2].set_xlim(-1,maxK)

    fig.suptitle(f"K = {bins.shape[0]}")

    plt.tight_layout()

    return (fig, ax)


def plot_est_bins2(bins, ests, E1, E2, maxK = None, d = None):
    plt.rcParams['figure.figsize'] = [16, 8]
    #plt.rcParams['figure.figsize'] = [18, 12]
    if maxK is None:
        maxK = bins.shape[0]
    fig, axd = plt.subplot_mosaic([['a', 'b'],['c','d'],['e', 'e']])
    stem_height = 0.1
    axd['a'].stem(bins, np.ones(bins.shape[0])*stem_height)
    axd['a'].set_title("Expected Bin Rewards $\mathbf{\mu}_{i,j}$")
    axd['a'].set_ylim(0,0.5)
    axd['a'].set_xlim(0,10)


    # Estimated \mu_i
    mu_i = np.mean(bins)
    band_rv = norm(loc=mu_i, scale=1)
    plot_rv(band_rv, ax=axd['b'])
    axd['b'].hist(bins, density=True, color = 'b', alpha = 0.5)
    axd['b'].set_title("Histogram of Expected Bin rewards")
    axd['b'].set_xlim(0, 10)
    axd['b'].set_ylim(0,1)

    # For ylims
    minY = np.min([np.min(E1),np.min(E2)]) - 0.1
    maxY =  np.max([np.max(E1), np.max(E2)]) + 0.1



    axd['c'].stem(ests, np.ones(bins.shape[0])*stem_height, markerfmt = 'go', linefmt = 'g')
    axd['c'].set_title("MLE Estimates of Bin Rewards $\hat{\mathbf{\mu}}_{i,j}$")
    axd['c'].set_ylim(0,0.5)
    axd['c'].set_xlim(0,10)



    est_mu_i = np.mean(ests)
    est_band_rv = norm(loc=est_mu_i, scale = 1)
    plot_rv(est_band_rv,ax = axd['d'], color = 'g')
    axd['d'].hist(ests, density = True, color = 'g', alpha = 0.5 )
    axd['d'].set_title("Histogram of Estimated Expected Bin rewards")
    axd['d'].set_xlim(0,10)
    axd['d'].set_ylim(0,1)

    # E1_mod = np.pad(E1,(maxK-E1.shape[0]), mode = 'empty')
    temp = np.empty(maxK - E1.shape[0])
    temp[:] = np.nan
    # print(temp)
    E1_mod = np.concatenate((E1, temp))
    # print(f"E1_mod: {E1_mod}")
    axd['e'].scatter(range(maxK), E1_mod, label = 'E1')
    axd['e'].set_ylim(minY, maxY)
    axd['e'].set_xlim(-1, maxK)
    axd['e'].set_title(f"E1(K)={round(E1[-1], 3)}, E2(K)={round(E2[-1],3)}")
    E2_mod = np.concatenate((E2, temp))
    axd['e'].scatter(range(maxK),E2_mod, color = 'g', label = "E2")
    axd['e'].set_ylabel('$K_i$')
    axd['e'].legend()


    if d is not None:
        fig.suptitle(f"Number of bins sampled ($K_i$) = {bins.shape[0]}; Samples per bin (d) = {d} ")
    else:
        fig.suptitle(f"Number of bins sampled ($K_i$) = {bins.shape[0]}")
    plt.tight_layout()

    return (fig, axd)


def plot_est_bins_an(Bins, Ests, E1, E2, d = None, skip = 1):
    K = Bins.shape[0]
    i = 0
    while i < K:
        bins = Bins[0:i+1]
        ests = Ests[0:i+1]
        e1 = E1[0:i+1]
        e2 = E2[0:i+1]
        #print(f"{(bins,ests)}")
        plot_est_bins2(bins, ests, e1, e2, maxK= K, d = d)
        i+=skip
    if i != K+skip:
        bins = Bins
        ests = Ests
        e1 = E1
        e2 = E2
        # print(f"{(bins,ests)}")
        plot_est_bins2(bins, ests, e1, e2, maxK=K, d=d)

