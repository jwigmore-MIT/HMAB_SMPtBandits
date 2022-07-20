from scipy.stats import norm
import numpy as np
from config import DEBUGSETTINGS as DS
from copy import deepcopy as dc

class Policy(object):

    def __init__(self, estimator, name = None):
        self.estimator = dc(estimator)
        self.name = name
        self.nbBands = estimator.nbBands
        self.nbBins = estimator.nbBins
        self.id = None
        self.rng = estimator.rng

    def __str__(self):
        return f"Policy: " + self.name

class HierarchicalThompsonSampling(Policy):

    def __init__(self, estimator, band_dec = 'order_stat', threshold = 0, initial_samples = 2, name = None, exp_frac = 0, stoch_dom = None):
        super().__init__(estimator,  name)
        self.estimator = estimator
        self.band_dec = band_dec
        self.threshold = threshold
        self.initial_samples = initial_samples
        self.exp_frac = exp_frac

    def order_stat_ub(self, mu, sigma, m):
        # From parameter samples mu and sigma, estimates the upper bound of the max of m sample from N(mu, sigma**2)
        return mu + sigma*np.sqrt(2*np.log(m))

    def choice(self, t = None):
        if t < self.initial_samples*self.nbBands:
            band = t % self.nbBands
            bin = self.rng.integers(0,self.nbBins)
        else:
                band, mu_i = self.choose_band()
                bin, q_val = self.choose_bin(band, mu_i)

        return (band, bin, 1)

    def choose_band(self):
        # Choose an arm by sampling from all band posteriors
        (mu_i, sigma_i) = self.estimator.sample_band_instances() # sigma_i is the known band variances if known
        q_i = np.zeros(self.nbBands)
        if self.band_dec == 'order_stat':
            for ii in range(self.nbBands):
                q_i[ii] = self.order_stat_ub(mu_i[ii], sigma_i[ii], self.nbBins)
        elif self.band_dec == 'threshold':
            for ii in range(self.nbBands):
                q_i[ii] = 1- norm.cdf(self.threshold, loc = mu_i[ii], scale = min(np.sqrt(sigma_i[ii]), 10000))
        elif self.band_dec == 'best_bin':
            # Sample mu_ij from the current best channel for each band
            mu_ij_samples, nu_ij_samples, tau_ij_samples = self.estimator.sample_best_bins(mu_i, sigma_i)
            q_i = mu_ij_samples
        band = q_i.argmax()

        return band, mu_i[band]

    def choose_bin(self, band_index: int, mu_i: float):
        '''
        Samples conditional bin posteriors for the given index and band posterior sample
        :param band_index: index of chosen band in which channels will lie
        :param mu_i:
        :return: (bin: int (chosen bin index), q_val: float (sampled expected reward for bin))
        '''
        q_ij = np.zeros(self.nbBins)
        mu_ij_samples, nu_ij, tau_ij = self.estimator.sample_bin_instances(band_index, mu_i)
        explore = False
        if self.exp_frac > 0:
            #var, mu, sample = self.estimator.get_latest_params()
            epsilon = self.exp_frac # should be function of var
            r = self.rng.random()
            if epsilon > r: explore = True
        if explore:
            bin = self.rng.integers(0, self.nbBins)
            q_val = None
        else:
            bin = mu_ij_samples.argmax()
            q_val = q_ij[bin]

        if DS["choose_bin"]:
            print(f"-" * 20)
            print(f"Random sample from Band ({band_index})'s posterior: {mu_i:.2f}")
            print(f"Bin conditional posterior parameters")
            np.set_printoptions(precision=2)
            print(nu_ij)
            print(tau_ij)
            print(f"Bin random samples")
            print(q_ij)

        return bin, q_val

class TwoStageHierarchicalThompsonSampling(Policy):
    '''Chooses the best band by sampling \hat\mu_i(t) ~ P(\mu_i(t)) and then \hat\mu_ij(t)P(\mu_ij(t)|\hat\mu_i(t))'''

    def __init__(self, estimator, band_dec = 'order_stat', threshold = 0, initial_samples = 2, name = None, exp_frac = 0, stoch_dom = None):
        super().__init__(estimator,  name)
        self.estimator = estimator
        # self.band_dec = band_dec # Most likely not need
        self.threshold = threshold
        self.initial_samples = initial_samples
        self.exp_frac = exp_frac

    def order_stat_ub(self, mu, sigma, m):
        # From parameter samples mu and sigma, estimates the upper bound of the max of m sample from N(mu, sigma**2)
        return mu + sigma*np.sqrt(2*np.log(m))

    def choice(self, t = None):
        if t < self.initial_samples*self.nbBands:
            band = t % self.nbBands
            bin = self.rng.integers(0,self.nbBins)
        else:
                band, mu_i = self.choose_band()
                bin, q_val = self.choose_bin(band, mu_i)

        return (band, bin, 1)

    def choose_band(self):
        # Choose an arm by sampling from all band posteriors
        (mu_i, sigma_i) = self.estimator.sample_band_instances()
        q_i = np.zeros(self.nbBands)
        if self.band_dec == 'order_stat':
            for ii in range(self.nbBands):
                q_i[ii] = self.order_stat_ub(mu_i[ii], sigma_i[ii], self.nbBins)
        elif self.band_dec == 'threshold':
            for ii in range(self.nbBands):
                q_i[ii] = 1- norm.cdf(self.threshold, loc = mu_i[ii], scale = min(np.sqrt(sigma_i[ii]), 10000))
        band = q_i.argmax()

        return band, mu_i[band]

    def choose_bin(self, band_index, mu_i, debug = False):
        q_ij = np.zeros(self.nbBins)
        (nu_ij, tau_ij) = self.estimator.sample_bin_instances(band_index, mu_i)
        explore = False
        if self.exp_frac > 0:
            #var, mu, sample = self.estimator.get_latest_params()
            epsilon = self.exp_frac # should be function of var
            r = self.rng.random()
            if epsilon > r: explore = True
        if explore:
            bin = self.rng.integers(0, self.nbBins)
            q_val = None
        else:
            for k in range(self.nbBins):
                q_ij[k] = norm.rvs(loc= nu_ij[k], scale = tau_ij[k]**(-1/2), random_state= self.rng)
                #q_ij[k] = 1-norm.cdf(self.threshold, loc = nu_ij[k], scale = tau_ij[k]**(-1/2))

            bin = q_ij.argmax()
            q_val = q_ij[bin]
        if DS["choose_bin"]:
            print(f"-" * 20)
            print(f"Random sample from Band ({band_index})'s posterior: {mu_i:.2f}")
            print(f"Bin conditional posterior parameters")
            np.set_printoptions(precision=2)
            print(nu_ij)
            print(tau_ij)
            print(f"Bin random samples")
            print(q_ij)
        return bin, q_val


class ChooseOne(Policy):
    def __init__(self, estimator, name = None, band = 0, bin = 0, samples = 1):
        super().__init__(estimator, name)
        self.samples = samples

    def choice(self,t):
        return self.band, self.bin, self.samples


class ChooseRandom(Policy):
    def __init__(self, estimator, samples = 1, name = None):
        super().__init__(estimator, name)
        self.samples = samples

    def choice(self,t):
        return np.random.randint(0,self.nbBands), np.random.randint(0, self.nbBins), self.samples


class UCB1(Policy):
    '''
    Classical UCB Implementation
    '''


    def choice(self, t = None):
        max_bb_index, max_arm_index, max_ucb = self.estimator.getMaxIndex()
        return max_bb_index + (1,) # +(1,) for returning the number of samples


class ChooseKUCB1(Policy):
    """
    Chooses K arms uniformly at random from each cluster
    """
    def __init__(self, K, ucb_estimator, name=None):
        super(UCB1, self).__init__(ucb_estimator, name)
        self.sampled_arms = []
        if K < self.nbBins: self.K = K
        else: print("The number of sampled arms must be less than nbBins")
        for b in range(self.nbBands):
            arm_indices = self.rng.integers(high = self.nbBins, size = K)
            self.sampled_arms.append(arm_indices)
            # Makes UCB index of each arm not sampled negative infinity
            for arm in range(self.nbBins):
                if arm not in arm_indices:
                    self.estimator.index[b,arm] = -np.infty



    def choice(self, t= None):
        pass







# class HierarchicalThompsonSampling2(Policy):
#
#     def __init__(self, estimator, nbBands, nbBins, init_threshold, name = None):
#         super().__init__(estimator, nbBands, nbBins)
#         self.estimator = estimator
#         self.nbBands = nbBands
#         self.nbBins = nbBins
#         self.threshold = init_threshold
#
#     def choice(self):
#         band, mu_i = self.choose_band()
#         bin, q_val = self.choose_bin(band, mu_i)
#
#         return (band, bin, 1)
#
#     def choose_band(self):
#         # Choose an arm by sampling from all band posteriors
#         (mu_i, sigma_i) = self.estimator.sample_band_instances()
#         # q_i = np.zeros(self.nbBands)
#         # for ii in range(self.nbBands):
#         #     q_i[ii] = 1- norm.cdf(self.threshold, loc = mu_i[ii], scale = sigma_i[ii])
#         # band = q_i.argmax()
#         band = mu_i.argmax()
#         return band, mu_i[band]
#
#     def choose_bin(self, band_index, mu_i, debug = False):
#         q_ij = np.zeros(self.nbBins)
#         (nu_ij, tau_ij) = self.estimator.sample_bin_instances(band_index, mu_i)
#         for k in range(self.nbBins):
#             q_ij[k] = norm.rvs(loc= nu_ij[k], scale = tau_ij[k]**(-1/2))
#             #q_ij[k] = 1-norm.cdf(self.threshold, loc = nu_ij[k], scale = tau_ij[k]**(-1/2))
#         if DS["choose_bin"]:
#             print(f"-"*20)
#             print(f"Random sample from Band ({band_index})'s posterior: {mu_i:.2f}")
#             print(f"Bin conditional posterior parameters")
#             np.set_printoptions(precision= 2)
#             print(nu_ij)
#             print(tau_ij)
#             print(f"Bin random samples")
#             print(q_ij)
#
#         bin = q_ij.argmax()
#         return bin, q_ij[bin]
