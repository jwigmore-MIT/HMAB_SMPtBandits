import numpy as np
from scipy.stats import norm, t
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

class Environment(object):


    def __init__(self, scenario):
        self.nbBands = scenario["nbBands"]      # Number of bands
        self.nbBins = scenario["nbBins"]        # Number of bins
        self.band_means = scenario["band_means"] # Generative band mean for each band
        self.band_variances = scenario["band_variances"]
        self.reward_variance = scenario["reward_variance"]  # Currently only the bin variance
        self.rng = scenario["rng"]
        self.bands = []
        self.bins = []
        self.bin_means = []
        self.best_bins = []
        self.best_means = []
        self.init_bands()

    def init_bands(self):
        for b in range(self.nbBands):
            self.bands.append(self.Band(self, b, self.nbBins, self.band_means[b], self.band_variances[b], self.reward_variance))
            self.bin_means.append(self.bands[b].bin_means) # The initialization of the band will generate all mean bin reward values
        for bin_list in self.bin_means:
            self.best_bins.append(np.argmax(bin_list))
            self.best_means.append(np.max(bin_list))
            x = 1


    def draw_samples(self, band_index, bin_index, k=1):
        return self.bands[band_index].bins[bin_index].sample(k)


    # <editor-fold desc="Plotting Methods">
    def plot_bands_dists(self, fig = None, ax = None, gen = True, samp = True, threshold = None):
        if fig is None:
            fig, ax = plt.subplots()
        for band in self.bands:
            band.plot_dists(fig, ax, gen, samp)
        if threshold is not None:
            ax.vlines(threshold, 0, ax.get_ylim()[1], colors = "black", linestyle = 'dashed', label = f"$q^*$ = {threshold}")
        ax.legend()
        fig.show()
    # </editor-fold>

    def compute_tails(self,threshold):
        if threshold is None:
            print("No Input threshold - No tails computed")
            return
        tails = []
        for b in range(self.nbBands):
            tails.append(self.bands[b].compute_tail(threshold))
        self.tail_mass = tails

    class Band(object):

        def __init__(self, Environment, band_index, nbBins, band_mean, band_var, bin_params):
            self.Environment = Environment
            self.index = band_index
            self.nbBins = nbBins
            self.mu = band_mean
            self.var = band_var
            self.sigma = np.sqrt(self.var)
            self.bin_params = bin_params
            self.rv = norm(loc = self.mu, scale = self.sigma)
            self.bins = []
            self.init_bins()
            self.id = None

        def init_bins(self):
            bin_means = self.rv.rvs(size = self.nbBins, random_state = self.Environment.rng)
            for b in range(self.nbBins):
                self.bins.append(Environment.Bin(self, b, bin_means[b], self.bin_params))
            self.sort_index = np.argsort(bin_means)
            self.sample_mu = np.mean(bin_means)
            self.sample_sigma = np.std(bin_means)
            self.bin_means = bin_means

        # <editor-fold desc="STR METHODS">
        def __str__(self):
            return f"""Band {self.index} ~ N({self.mu:.1f}, {self.sigma**2:.2f})/N({self.sample_mu:.1f}, {self.sample_sigma**2:.2f})"""

        def __genstr__(self):
            return f"""Band {self.index} (G) ~ N({self.mu:.1f}, {self.sigma**2:.2f})"""

        def __sampstr__(self):
            return f"""Band {self.index} (S) ~ N({self.sample_mu:.1f}, {self.sample_sigma**2:.2f})"""
        def __sampstr2__(self):
            return f"""N({self.sample_mu:.1f}, {self.sample_sigma**2:.2f})"""
        # </editor-fold>

        # <editor-fold desc="PLOTTING METHODS">
        def plot_dists(self, fig = None, ax = None, gen = True, samp = True):
            # Plots both the generative and sample distributions of the Band
            plot = False
            if fig is None:
                fig, ax = plt.subplots()
                plot = True
            ax.set_title("[Environment] Band Generative and Sample Distributions")
            if gen:
                self.plot_gen_dist(fig, ax)
            if samp:
                self.plot_samp_dist(fig, ax)
            if plot:
                ax.legend()
                fig.show()


        def plot_gen_dist(self, fig = None, ax = None):
            if fig is None:
                fig, ax = plt.subplots()
            x = np.linspace(self.rv.ppf(0.001), self.rv.ppf(.999))
            pdf = self.rv.pdf(x)
            ax.plot(x, pdf,label = self.__genstr__())
            return fig, ax

        def plot_samp_dist(self, fig = None, ax = None):
            if fig is None:
                fig, ax = plt.subplots()
            s_rv = norm(loc = self.sample_mu, scale = self.sample_sigma)
            x = np.linspace(s_rv.ppf(0.001), s_rv.ppf(.999))
            pdf = s_rv.pdf(x)
            ax.plot(x, pdf,'-', label=self.__sampstr__())
            return fig, ax
        # </editor-fold>

        def compute_tail(self,threshold):
            return 1 - norm.cdf(threshold, loc = self.sample_mu, scale = self.sample_sigma)

    class Bin(object):

        def __init__(self, Band, bin_index, mu, sigma):
            self.Band = Band
            self.index = bin_index
            self.mu = mu
            self.sigma = sigma

        def sample(self, k =1):
            return norm.rvs(loc = self.mu, scale = self.sigma, size = k, random_state=self.Band.Environment.rng)

        def __str__(self):
            return f"""Bin{self.index} ~ N({self.mu:.1f}, {self.sigma**2:.2f})"""

