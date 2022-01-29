# Local  Imports
from .Distribution import Distribution

# Imports
from scipy.stats import norm
import numpy as np
from numpy.random import default_rng
rng = default_rng()


class Normal(Distribution):

    def __init__(self, mu, sigma):
        self._mu = mu
        self.mu = mu
        self._sigma = sigma
        self.sigma = sigma

    def __str__(self, init = False):
        if init:
            return "Normal(_mu = {:.3g},_sigma = {:.3g})".format(self._mu, self._sigma)
        else:
            return "Normal(mu = {:.3g},sigma = {:.3g})".format(self.mu, self.sigma)


    def reset(self, *args):
        """Reset the parameters of the distribution. If arguments are passed replace old params with arguments"""
        if not args:
            self.mu = self._mu
            self.sigma = self._sigma
        else:
            self.mu = args[0]
            self.sigma = args[1]

    def sample(self):
        return rng.normal(loc = self.mu, scale = self.sigma)

    def sample_k(self, k = 100):
        return rng.normal(loc = self.mu, scale = self.sigma,size = k)

    def getInitParams(self): # Return the initial parameters
        return self.mu, self._sigma

    def getVariance(self):
        return self.sigma**2

    def getExpVal(self):
        return self.mu

    def plot(self, ax, label = None):
        a = .005 # Quantiles for plotting
        points = 1000 # Number of points to plot
        rv = norm(self.mu, self.sigma)
        x = np.linspace(rv.ppf(a), rv.ppf(1-a), points)
        if label is None:
            label = self.__str__()
        ax.plot(x,rv.pdf(x), label =label)
        return ax



