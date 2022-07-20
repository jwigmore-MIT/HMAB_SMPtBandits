from typing import Union

import numpy as np
from scipy.stats import norm, invgamma
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from analysistools import pr_max_X, sim_pr_max
from config import DEBUGSETTINGS as DS
np.seterr(all = "raise")

def get_arm_index(band_index, bin_index, nbBands, nbBins):
    return np.ravel_multi_index((band_index, bin_index), (nbBands, nbBins))


def get_bb_index(arm_index, nbBands, nbBins):
    return np.unravel_index(arm_index, (nbBands, nbBins))


class ModalBayesEstimator(object):
    # Factors in estimates of the band variance into the posterior distribution of band means
    def __init__(self, environment, percentile=0.95):
        self.nbBands = environment.nbBands
        self.nbBins = environment.nbBins
        self.bands = []
        self.history = {}
        self.true_band_vars = environment.band_variances  # True band variance (only used for debugging)
        self.reward_variance = environment.reward_variance  # Known bin reward variance
        self.var_method = 0
        self.id = None
        self.percentile = percentile  # Confidence Interval Percentile
        self.parameter_strings = ["eta", "df", "sigma", "rho", "nu", "theta"]
        self.rng = environment.rng
        # Initialize Bands (and bins)
        for b in range(self.nbBands):
            self.bands.append(self.Band(self, b))

    def update(self, t, band_index, bin_index, samples):
        self.bands[band_index].update(bin_index, samples, t)

    def get_latest_params(self):
        latest_params = np.zeros([6, self.nbBands])
        for b in range(self.nbBands):
            latest_params[0,b] = self.bands[b].eta[-1]
            latest_params[1,b] = self.bands[b].df[-1]
            latest_params[2,b] = self.bands[b].sigma_i[-1]
            latest_params[3,b] = self.bands[b].tau[-1]**(-1)
            latest_params[4,b] = self.bands[b].nu[-1]
            latest_params[5,b] = self.bands[b].theta[-1]
        return latest_params


    def sample_band_instances(self):
        mu_i_samples = np.zeros(self.nbBands)
        var_i_samples = np.zeros(self.nbBands)
        for k in range(self.nbBands):
            var_i_samples[k] = self.bands[k].sample_var_posterior()
            mu_i_samples[k] = self.bands[k].sample_mean_posterior(var_i_samples[k]) # Sample from the conditional posterior for the band mean
        return mu_i_samples, var_i_samples

    def sample_bin_instances(self, band_index, mu_i):
        band = self.bands[band_index]
        nu_ij_samples = np.zeros(self.nbBins)
        tau_ij_samples = np.zeros(self.nbBins)
        for k in range(self.nbBins):
            (nu_ij_samples[k], tau_ij_samples[k]) = band.bins[k].compute_posterior_parameters(mu_i)
        return nu_ij_samples, tau_ij_samples

    def print_band_parameter_estimates(self, band_index):
        for b in band_index:
            print(self.bands[b].get_posterior_str())

    class Band:
        # Store and update the belief for the band parameters
        '''
        The band parameters are nu and tau_i, and we store it after each update in list fashion
        '''

        def __init__(self, estimator, band_index):
            self.estimator = estimator
            self.band_index = band_index
            self.nbBins = estimator.nbBins
            self.true_band_var = estimator.true_band_vars[band_index]
            self.est_band_var = [np.infty]  #
            self.reward_variance = estimator.reward_variance
            # self.uninformative = True  # We start with a non-informative prior over the mean
            self.pulls = 0
            self.history = {
                "Time": [],
                "Bin": [],
                "Reward": []
            }
            self.bins = []
            self.obs_bins = []

            self.eta = [np.nan]  # Scale parameter for band variance posterior
            self.df = [np.nan] # Degrees freedom parameter for band variance posterior
            self.sigma_i = [np.nan] # Samples from band variance posterior

            self.nu = [np.nan]  # Location parameter for the conditional band mean posterior
            self.tau = [np.nan]  # Scale parameter for the band mean posterior

            self.theta = [np.nan]


            for b in range(self.nbBins):
                self.bins.append(ModalBayesEstimator.Bin(self, b))

        def update(self, bin, sample, time):
            self.bins[bin].add_observations(sample)
            self.add_observations(time, bin, sample)
            self.update_variance_posterior()


        def add_observations(self, time, bin, samples):
            self.history["Time"].append(time)
            self.history["Bin"].append(bin)
            self.history["Reward"].append(samples)


        def update_variance_posterior(self):
            K_i = len(self.obs_bins) # number of observed bins
            if K_i < 2:
                self.eta.append(np.nan)
            else:
                sample_means = [self.bins[bin].sample_mean for bin in self.obs_bins]
                self.eta.append(np.var(sample_means))
            self.df.append(K_i-1)
            return self.eta[-1], self.df[-1]

        def sample_var_posterior(self, k=1):
            if self.df[-1] < 1: # Need non-zero degrees of freedom to have valid pdf
                self.sigma_i[-1] = self.rng.rand(1)*100
            else:
                self.sigma_i[-1] = invgamma.rvs(a=self.df[-1] / 2, scale=self.df[-1] * self.eta[-1] / 2, random_state= self.estimator.rng)
            return self.sigma_i[-1]


        def sample_mean_posterior(self, sigma_i = None):
            if sigma_i is None:
                sigma_i = self.sigma_i[-1]
            if np.isnan(sigma_i):
                tau = np.nan
                nu = np.nan
                theta = np.nan
            else:
                tau = 0
                numerator = 0
                for bin in self.obs_bins:
                    bin = self.bins[bin]
                    obs = bin.observations
                    s_ij = bin.reward_variance / len(obs)
                    tau += 1/(s_ij+self.sigma_i[-1])
                    numerator += 1/(s_ij+self.sigma_i[-1])*np.mean(obs)
                nu = numerator/tau
                theta = norm.rvs(loc = nu, scale = tau**(-1/2), random_state= self.estimator.rng)
            self.theta[-1] = theta
            self.nu[-1] = nu
            self.tau[-1] = tau
            return theta

        def compute_CI(self, p):
            self.CI = [p, norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(1 - p),
                       norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(p)]
            return self.CI



        def get_prior_str(self):

            if self.tau[-2] > 0:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, {self.tau[-2] ** (-1):.2f})"
            else:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, \\infty)"

        def get_mean_str(self):
            return f"Posterior for mu ({self.band_index}): N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_mean_posterior_str(self):
            return f"N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_band_variance_est(self):
            return f"{self.est_band_var[-1]:.2f}"

        def get_variance_posterior_str(self):
            return f"Inx-X^2({self.df[-1]}, {self.eta[-1]:.2f})"


        def plot_posterior_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            ax.plot(x, self.nu[1:], label=f"B({self.band_index}) Posterior Mean")
            CI = 2 * np.power(self.tau, np.ones(len(
                self.tau)) * -1 / 2)  # Compute the posterior standard deviation and take 2 times this value as the 95% credible interval
            upper_CI = self.nu + CI
            lower_CI = self.nu - CI
            ax.fill_between(x, lower_CI[1:], upper_CI[1:], alpha=0.1, label="95% Credible Interval")

            # ax.plot(x, upper_CI[1:])
            # ax.plot(x, lower_CI[1:])
            ax.legend()

        def plot_posterior_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            var = np.power(self.tau, np.ones(len(self.tau)) * -1)
            ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
            ax.legend()

        def plot_band_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            var = np.power(self.sigma_i_est, np.ones(len(self.sigma_i_est)) * -1)
            ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
            ax.legend()

        # <editor-fold desc="Unit Tests">
        # Unit Tests
        def sample_posterior_unit_test(self, k=10000):
            samples = self.sample_posterior(k)
            sample_mean = np.mean(samples)
            sample_scale = np.std(samples)
            print(f"The sample scale for the {k} samples of the posterior is $\hat\sigma$ = {sample_scale}")
        # </editor-fold>

    class Bin:
        # Store and update the belief for the band parameter (only for mean)
        def __init__(self, band, bin_index):
            self.band = band
            self.index = bin_index
            self.reward_variance = band.reward_variance
            self.true_band_var = band.true_band_var
            self.sample_mean = None
            self.ss = 0
            self.sample_var = None
            self.nu = None
            self.tau = 1 / self.reward_variance
            self.observations = []

        def add_observations(self, observations):
            self.observations.extend(observations)
            self.compute_mean(observations)
            if len(self.observations) == 1:
                self.band.obs_bins.append(self.index)

        def compute_mean(self, observation):
            if self.sample_mean is None:  # first sample of this bin
                self.sample_mean = observation
            else:  # recursive mean calcluation
                n = len(self.observations)
                self.sample_mean = 1 / n * ((n - 1) * self.sample_mean + observation)

        def compute_posterior_parameters(self, mu_i):

            nb_obs = len(self.observations)
            if nb_obs > 0:  # If we have no observations,
                s_ij = self.reward_variance / nb_obs
                obs_mean = np.mean(self.observations)
                precision = 1 / s_ij + 1 / self.band.sigma_i[-1]
                numerator = 1 / s_ij * obs_mean + 1 / (self.band.sigma_i[-1]) * mu_i
                posterior_mean = numerator / precision
            else:
                precision = 1 / self.band.sigma_i[-1]
                posterior_mean = mu_i

            return posterior_mean, precision


class PointVarianceEstimator(object):
    # Factors in estimates of the band variance into the posterior distribution of band means
    def __init__(self, environment, percentile=0.95):
        self.nbBands = environment.nbBands
        self.nbBins = environment.nbBins
        self.bands = []
        self.history = {}
        self.true_band_vars = environment.band_variances  # True band variance (only used for debugging)
        self.reward_variance = environment.reward_variance  # Known bin reward variance
        self.var_method = 0
        self.id = None
        self.percentile = percentile  # Confidence Interval Percentile
        self.parameter_strings = ["eta", "df", "sigma", "rho", "nu", "theta"]
        self.rng = environment.rng
        # Initialize Bands (and bins)
        for b in range(self.nbBands):
            self.bands.append(self.Band(self, b))

    def update(self, t, band_index, bin_index, samples):
        self.bands[band_index].update(bin_index, samples, t)

    def get_latest_params(self):
        latest_params = np.zeros([6, self.nbBands])
        for b in range(self.nbBands):
            latest_params[0,b] = self.bands[b].eta[-1]
            latest_params[1,b] = self.bands[b].df[-1]
            latest_params[2,b] = self.bands[b].sigma_i[-1]
            latest_params[3,b] = self.bands[b].tau[-1]**(-1)
            latest_params[4,b] = self.bands[b].nu[-1]
            latest_params[5,b] = self.bands[b].theta[-1]
        return latest_params


    def sample_band_instances(self):
        mu_i_samples = np.zeros(self.nbBands)
        var_i_samples = np.zeros(self.nbBands)
        for k in range(self.nbBands):
            var_i_samples[k] = self.bands[k].sample_var_posterior()
            mu_i_samples[k] = self.bands[k].sample_mean_posterior(var_i_samples[k]) # Sample from the conditional posterior for the band mean
        return mu_i_samples, var_i_samples

    def sample_bin_instances(self, band_index, mu_i):
        band = self.bands[band_index]
        nu_ij_samples = np.zeros(self.nbBins)
        tau_ij_samples = np.zeros(self.nbBins)
        for k in range(self.nbBins):
            (nu_ij_samples[k], tau_ij_samples[k]) = band.bins[k].compute_posterior_parameters(mu_i)
        return nu_ij_samples, tau_ij_samples

    def print_band_parameter_estimates(self, band_index):
        for b in band_index:
            print(self.bands[b].get_posterior_str())

    class Band:
        # Store and update the belief for the band parameters
        '''
        The band parameters are nu and tau_i, and we store it after each update in list fashion
        '''

        def __init__(self, estimator, band_index):
            self.estimator = estimator
            self.band_index = band_index
            self.nbBins = estimator.nbBins
            self.true_band_var = estimator.true_band_vars[band_index]
            self.est_band_var = [np.infty]  #
            self.reward_variance = estimator.reward_variance
            # self.uninformative = True  # We start with a non-informative prior over the mean
            self.pulls = 0
            self.history = {
                "Time": [],
                "Bin": [],
                "Reward": []
            }
            self.bins = []
            self.obs_bins = []

            self.eta = [np.nan]  # Scale parameter for band variance posterior
            self.df = [np.nan] # Degrees freedom parameter for band variance posterior
            self.sigma_i = [np.nan] # Samples from band variance posterior

            self.nu = [np.nan]  # Location parameter for the conditional band mean posterior
            self.tau = [np.nan]  # Scale parameter for the band mean posterior

            self.theta = [np.nan]


            for b in range(self.nbBins):
                self.bins.append(ModalBayesEstimator.Bin(self, b))

        def update(self, bin, sample, time):
            self.bins[bin].add_observations(sample)
            self.add_observations(time, bin, sample)
            self.update_variance_posterior()


        def add_observations(self, time, bin, samples):
            self.history["Time"].append(time)
            self.history["Bin"].append(bin)
            self.history["Reward"].append(samples)


        def update_variance_posterior(self):
            K_i = len(self.obs_bins) # number of observed bins
            if K_i < 2:
                self.eta.append(np.nan)
            else:
                sample_means = [self.bins[bin].sample_mean for bin in self.obs_bins]
                self.eta.append(np.var(sample_means))
            self.df.append(self.df[-1])
            return self.eta[-1], self.df[-1]

        def sample_var_posterior(self, k=1):
            # if self.df[-1] < 1: # Need non-zero degrees of freedom to have valid pdf
            #     self.sigma_i[-1] = self.rng.rand(1)*100
            # else:
            #     self.sigma_i[-1] = invgamma.rvs(a=self.df[-1] / 2, scale=self.df[-1] * self.eta[-1] / 2, random_state= self.estimator.rng)
            self.sigma_i[-1] = self.eta[-1]
            return self.sigma_i[-1]


        def sample_mean_posterior(self, sigma_i = None):
            if sigma_i is None:
                sigma_i = self.sigma_i[-1]
            if np.isnan(sigma_i):
                tau = np.nan
                nu = np.nan
                theta = np.nan
            else:
                tau = 0
                numerator = 0
                for bin in self.obs_bins:
                    bin = self.bins[bin]
                    obs = bin.observations
                    s_ij = bin.reward_variance / len(obs)
                    tau += 1/(s_ij+self.sigma_i[-1])
                    numerator += 1/(s_ij+self.sigma_i[-1])*np.mean(obs)
                nu = numerator/tau
                theta = norm.rvs(loc = nu, scale = tau**(-1/2), random_state= self.estimator.rng)
            self.theta[-1] = theta
            self.nu[-1] = nu
            self.tau[-1] = tau
            return theta


        def get_prior_str(self):

            if self.tau[-2] > 0:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, {self.tau[-2] ** (-1):.2f})"
            else:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, \\infty)"

        def get_mean_str(self):
            return f"Posterior for mu ({self.band_index}): N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_mean_posterior_str(self):
            return f"N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_band_variance_est(self):
            return f"{self.est_band_var[-1]:.2f}"

        def get_variance_posterior_str(self):
            return f"Inx-X^2({self.df[-1]}, {self.eta[-1]:.2f})"


    class Bin:
        # Store and update the belief for the band parameter (only for mean)
        def __init__(self, band, bin_index):
            self.band = band
            self.index = bin_index
            self.reward_variance = band.reward_variance
            self.true_band_var = band.true_band_var
            self.sample_mean = None
            self.ss = 0
            self.sample_var = None
            self.nu = None
            self.tau = 1 / self.reward_variance
            self.observations = []

        def add_observations(self, observations):
            self.observations.extend(observations)
            self.compute_mean(observations)
            if len(self.observations) == 1:
                self.band.obs_bins.append(self.index)

        def compute_mean(self, observation):
            if self.sample_mean is None:  # first sample of this bin
                self.sample_mean = observation
            else:  # recursive mean calcluation
                n = len(self.observations)
                self.sample_mean = 1 / n * ((n - 1) * self.sample_mean + observation)

        def compute_posterior_parameters(self, mu_i):

            nb_obs = len(self.observations)
            if nb_obs > 0:  # If we have no observations,
                s_ij = self.reward_variance / nb_obs
                obs_mean = np.mean(self.observations)
                precision = 1 / s_ij + 1 / self.band.sigma_i[-1]
                numerator = 1 / s_ij * obs_mean + 1 / (self.band.sigma_i[-1]) * mu_i
                posterior_mean = numerator / precision
            else:
                precision = 1 / self.band.sigma_i[-1]
                posterior_mean = mu_i

            return posterior_mean, precision


class EMBayesEstimator(object):
    # Factors in estimates of the band variance into the posterior distribution of band means
    def __init__(self, environment, percentile=0.95):
        self.nbBands = environment.nbBands
        self.nbBins = environment.nbBins
        self.bands = []
        self.history = {}
        self.true_band_vars = environment.band_variances  # True band variance (only used for debugging)
        self.reward_variance = environment.reward_variance  # Known bin reward variance
        self.id = None
        self.percentile = percentile  # Confidence Interval Percentile
        self.parameter_strings = ["eta", "df", "sigma", "rho", "nu", "theta"]
        self.rng = environment.rng
        # Initialize Bands (and bins)
        for b in range(self.nbBands):
            self.bands.append(self.Band(self, b))

    def update(self, t, band_index, bin_index, samples):
        self.bands[band_index].update(t, bin_index, samples)

    def get_latest_params(self):
        latest_params = np.zeros([6, self.nbBands])
        for b in range(self.nbBands):
            latest_params[0,b] = self.bands[b].eta[-1]
            latest_params[1,b] = self.bands[b].df[-1]
            latest_params[2,b] = self.bands[b].sigma_i[-1]
            latest_params[3,b] = self.bands[b].tau[-1]**(-1)
            latest_params[4,b] = self.bands[b].nu[-1]
            latest_params[5,b] = self.bands[b].theta[-1]
        return latest_params


    def sample_band_instances(self):
        mu_i_samples = np.zeros(self.nbBands)
        var_i_samples = np.zeros(self.nbBands)
        for k in range(self.nbBands):
            var_i_samples[k] = self.bands[k].sample_var_posterior()
            mu_i_samples[k] = self.bands[k].sample_mean_posterior(var_i_samples[k]) # Sample from the conditional posterior for the band mean
        return mu_i_samples, var_i_samples

    def sample_bin_instances(self, band_index, mu_i):
        band = self.bands[band_index]
        nu_ij_samples = np.zeros(self.nbBins)
        tau_ij_samples = np.zeros(self.nbBins)
        for k in range(self.nbBins):
            (nu_ij_samples[k], tau_ij_samples[k]) = band.bins[k].compute_posterior_parameters(mu_i)
        return nu_ij_samples, tau_ij_samples

    def print_band_parameter_estimates(self, band_index):
        for b in band_index:
            print(self.bands[b].get_posterior_str())

    class Band:
        # Store and update the belief for the band parameters
        '''
        The band parameters are nu and tau_i, and we store it after each update in list fashion
        '''

        def __init__(self, estimator, band_index):
            self.estimator = estimator
            self.band_index = band_index
            self.nbBins = estimator.nbBins
            self.true_band_var = estimator.true_band_vars[band_index]
            self.est_band_var = [np.infty]  #
            self.reward_variance = estimator.reward_variance
            # self.uninformative = True  # We start with a non-informative prior over the mean
            self.pulls = 0
            self.history = {
                "Time": [],
                "Bin": [],
                "Reward": []
            }
            self.bins = []
            self.obs_bins = []

            self.eta = [np.nan]  # Scale parameter for band variance posterior
            self.df = [np.nan] # Degrees freedom parameter for band variance posterior
            self.sigma_i = [np.nan] # Samples from band variance posterior

            self.nu = [np.nan]  # Location parameter for the conditional band mean posterior
            self.tau = [np.nan]  # Scale parameter for the band mean posterior

            self.theta = [np.nan]


            for b in range(self.nbBins):
                self.bins.append(ModalBayesEstimator.Bin(self, b))

        def update(self, bin, sample, time):
            self.bins[bin].add_observations(sample)
            self.add_observations(time, bin, sample)
            self.update_variance_posterior()


        def add_observations(self, time, bin, samples):
            self.history["Time"].append(time)
            self.history["Bin"].append(bin)
            self.history["Reward"].append(samples)


        def update_variance_posterior(self):
            K_i = len(self.obs_bins) # number of observed bins
            if K_i < 2:
                self.eta.append(np.nan)
            else:
                sample_means = [self.bins[bin].sample_mean for bin in self.obs_bins]
                self.eta.append(np.var(sample_means))
            self.df.append(K_i-1)
            return self.eta[-1], self.df[-1]

        def sample_var_posterior(self, k=1):
            if self.df[-1] < 1: # Need non-zero degrees of freedom to have valid pdf
                self.sigma_i[-1] = self.rng.rand(1)*100
            else:
                self.sigma_i[-1] = invgamma.rvs(a=self.df[-1] / 2, scale=self.df[-1] * self.eta[-1] / 2, random_state= self.estimator.rng)
            return self.sigma_i[-1]


        def sample_mean_posterior(self, sigma_i = None):
            if sigma_i is None:
                sigma_i = self.sigma_i[-1]
            if np.isnan(sigma_i):
                tau = np.nan
                nu = np.nan
                theta = np.nan
            else:
                tau = 0
                numerator = 0
                for bin in self.obs_bins:
                    bin = self.bins[bin]
                    obs = bin.observations
                    s_ij = bin.reward_variance / len(obs)
                    tau += 1/(s_ij+self.sigma_i[-1])
                    numerator += 1/(s_ij+self.sigma_i[-1])*np.mean(obs)
                nu = numerator/tau
                theta = norm.rvs(loc = nu, scale = tau**(-1/2), random_state= self.estimator.rng)
            self.theta[-1] = theta
            self.nu[-1] = nu
            self.tau[-1] = tau
            return theta

        def compute_CI(self, p):
            self.CI = [p, norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(1 - p),
                       norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(p)]
            return self.CI



        def get_prior_str(self):

            if self.tau[-2] > 0:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, {self.tau[-2] ** (-1):.2f})"
            else:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, \\infty)"

        def get_mean_str(self):
            return f"Posterior for mu ({self.band_index}): N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_mean_posterior_str(self):
            return f"N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_band_variance_est(self):
            return f"{self.est_band_var[-1]:.2f}"

        def get_variance_posterior_str(self):
            return f"Inx-X^2({self.df[-1]}, {self.eta[-1]:.2f})"


        def plot_posterior_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            ax.plot(x, self.nu[1:], label=f"B({self.band_index}) Posterior Mean")
            CI = 2 * np.power(self.tau, np.ones(len(
                self.tau)) * -1 / 2)  # Compute the posterior standard deviation and take 2 times this value as the 95% credible interval
            upper_CI = self.nu + CI
            lower_CI = self.nu - CI
            ax.fill_between(x, lower_CI[1:], upper_CI[1:], alpha=0.1, label="95% Credible Interval")

            # ax.plot(x, upper_CI[1:])
            # ax.plot(x, lower_CI[1:])
            ax.legend()

        def plot_posterior_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            var = np.power(self.tau, np.ones(len(self.tau)) * -1)
            ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
            ax.legend()

        def plot_band_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            var = np.power(self.sigma_i_est, np.ones(len(self.sigma_i_est)) * -1)
            ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
            ax.legend()

        # <editor-fold desc="Unit Tests">
        # Unit Tests
        def sample_posterior_unit_test(self, k=10000):
            samples = self.sample_posterior(k)
            sample_mean = np.mean(samples)
            sample_scale = np.std(samples)
            print(f"The sample scale for the {k} samples of the posterior is $\hat\sigma$ = {sample_scale}")
        # </editor-fold>

    class Bin:
        # Store and update the belief for the band parameter (only for mean)
        def __init__(self, band, bin_index):
            self.band = band
            self.index = bin_index
            self.reward_variance = band.reward_variance
            self.true_band_var = band.true_band_var
            self.sample_mean = None
            self.ss = 0
            self.sample_var = None
            self.nu = None
            self.tau = 1 / self.reward_variance
            self.observations = []

        def add_observations(self, observations):
            self.observations.extend(observations)
            self.compute_mean(observations)
            if len(self.observations) == 1:
                self.band.obs_bins.append(self.index)

        def compute_mean(self, observation):
            if self.sample_mean is None:  # first sample of this bin
                self.sample_mean = observation
            else:  # recursive mean calcluation
                n = len(self.observations)
                self.sample_mean = 1 / n * ((n - 1) * self.sample_mean + observation)

        def compute_posterior_parameters(self, mu_i):

            nb_obs = len(self.observations)
            if nb_obs > 0:  # If we have no observations,
                s_ij = self.reward_variance / nb_obs
                obs_mean = np.mean(self.observations)
                precision = 1 / s_ij + 1 / self.band.sigma_i[-1]
                numerator = 1 / s_ij * obs_mean + 1 / (self.band.sigma_i[-1]) * mu_i
                posterior_mean = numerator / precision
            else:
                precision = 1 / self.band.sigma_i[-1]
                posterior_mean = mu_i

            return posterior_mean, precision

class KnownVarianceEstimator(object):
    # Factors in estimates of the band variance into the posterior distribution of band means
    def __init__(self, environment, percentile=0.95):
        self.env = environment
        self.nbBands = environment.nbBands
        self.nbBins = environment.nbBins
        self.bands = []
        self.history = {}
        self.var_flag = False
        self.true_band_vars = environment.band_variances  #
        self.reward_variance = environment.reward_variance  # Known bin reward variance
        self.id = None
        r = percentile  # Confidence Interval Percentile
        self.parameter_strings = ["rho", "nu", "theta"]
        self.rng = environment.rng
        # Initialize Bands (and bins)
        for b in range(self.nbBands):
            self.bands.append(self.Band(self, b))


    def update(self, t:int, band_index: int, bin_index: int, rewards: Union[int,list]):
        """
        Updates the history from a(t) = (band_index, bin_index) with observed reward(s)
        :param t: round of observation
        :param band_index: chosen band
        :param bin_index: chosen bin
        :param rewards: observed reward(s)
        :return:
        """
        self.bands[band_index].update(t, bin_index, rewards)

    def get_latest_params(self):
        latest_params = np.zeros([3, self.nbBands])
        for b in range(self.nbBands):
            latest_params[0,b] = self.bands[b].tau[-1]**(-1)
            latest_params[1,b] = self.bands[b].nu[-1]
            latest_params[2,b] = self.bands[b].theta[-1]
        return latest_params


    def sample_band_instances(self):
        mu_i_samples = np.zeros(self.nbBands)
        for k in range(self.nbBands):
            mu_i_samples[k] = self.bands[k].sample_mean_posterior() # Sample from the conditional posterior for the band mean
        return mu_i_samples, self.true_band_vars

    def sample_best_bins(self, mu_i_samples: float, sigma_i_samples: float):
        """

        :param mu_i_samples: samples of mu_i for each band from their respective posteriors
        :param sigma_i_samples: true band variance (but this is retrieved in the bin.compute_posterior_parameter() so
               unused
        :return: [list, list, list]: lists of samples from posterior, posterior mean parameters, and
                                     posterior precision parameters
        """
        nu_ij_samples = np.zeros(self.nbBands)
        tau_ij_samples = np.zeros(self.nbBands)
        mu_ij_samples = np.zeros(self.nbBins)

        for k in range(self.nbBands):
            best_bin_index = self.bands[k].best_bin # index of best bin
            nu_ij_samples[k], tau_ij_samples[k] = self.bands[k].bins[best_bin_index].compute_posterior_parameters(mu_i_samples[k])
            mu_ij_samples[k] = norm.rvs(loc = nu_ij_samples[k], scale = tau_ij_samples[k]**(-1/2), random_state= self.rng)

        return mu_ij_samples, nu_ij_samples, tau_ij_samples



    def sample_bin_instances(self, band_index, mu_i):
        band = self.bands[band_index]
        nu_ij_samples = np.zeros(self.nbBins)
        tau_ij_samples = np.zeros(self.nbBins)
        mu_ij_samples = np.zeros(self.nbBins)
        for k in range(self.nbBins):
            (nu_ij_samples[k], tau_ij_samples[k]) = band.bins[k].compute_posterior_parameters(mu_i)
            mu_ij_samples[k] = norm.rvs(loc = nu_ij_samples[k], scale = tau_ij_samples[k]**(-1/2), random_state= self.rng)
        return mu_ij_samples, nu_ij_samples, tau_ij_samples



    def get_band_posterior_strs(self, band_index = "all", printit = False):
        if band_index == 'all':
            band_index = range(self.nbBands)
        post_strs = []
        for b in band_index:
            post_strs.append(self.bands[b].get_mean_str())
            if printit: print(post_strs[-1])
        return post_strs

    def test_posterior_dominance(self, delta = 0.01, gran = 1000):
        X = []
        for band in self.bands:
            if np.isnan(band.nu[-1]) or np.isnan(band.tau[-1]):
                return np.nan, np.nan, np.nan
            X_i = norm(loc = band.nu[-1], scale = band.tau[-1]**(-1/2))
            X.append(X_i)
        a_i = sim_pr_max(X, M = gran)

        best_band = np.argmax(a_i)
        is_dom = np.max(a_i)> 1- delta
        return best_band, is_dom, a_i



    class Band:
        # Store and update the belief for the band parameters
        '''
        The band parameters are nu and tau_i, and we store it after each update in list fashion
        '''

        def __init__(self, estimator, band_index):
            self.estimator = estimator
            self.band_index = band_index
            self.nbBins = estimator.nbBins
            self.true_band_var = estimator.true_band_vars[band_index]
            self.est_band_var = [np.infty]  #
            self.reward_variance = estimator.reward_variance
            # self.uninformative = True  # We start with a non-informative prior over the mean
            self.pulls = 0
            self.history = {
                "Time": [],
                "Bin": [],
                "Reward": []
            }
            self.bins = []
            self.obs_bins = []

            self.nu = [np.nan]  # Location parameter for the conditional band mean posterior
            self.tau = [np.nan]  # Scale parameter for the band mean posterior

            self.theta = [np.nan]

            self.best_bin = None
            self.best_bin_mu = -np.infty


            for b in range(self.nbBins):
                self.bins.append(KnownVarianceEstimator.Bin(self, b))

        def update(self, t: int, bin_index: int, rewards: Union[float, list]):
            """
                Updates the history from a(t) = (bin_index) with observed reward(s)
                :param t: round of observation
                :param bin_index: chosen bin
                :param rewards: observed reward(s)
                :return:
                """
            self.bins[bin_index].add_observations(rewards)
            self.add_observations(t, bin_index, rewards)


        def add_observations(self, time, bin, samples):
            self.history["Time"].append(time)
            self.history["Bin"].append(bin)
            self.history["Reward"].append(samples)


        def sample_mean_posterior(self):

            tau = 0
            numerator = 0
            for bin_index in self.obs_bins:
                bin = self.bins[bin_index]
                obs = bin.observations
                s_ij =  bin.reward_variance / len(obs)
                tau += 1 / (s_ij + self.true_band_var ** 2)
                numerator += np.mean(obs) / (s_ij + self.true_band_var ** 2)

            nu = numerator/tau
            self.nu.append(nu)
            self.tau.append(tau)
            theta = norm.rvs(loc = nu, scale = tau**(-1/2), random_state= self.estimator.rng)
            self.theta.append(theta)
            return theta

        def update_best_bin(self, index, bin_sample_mean):
            """
            Updates recorded best bin if computed bin_sample_mean is the max seen so far
            Gets called from child bin.add_observation

            :param index:
            :param bin_sample_mean:
            :return:
            """
            if bin_sample_mean < self.best_bin_mu:
                return
            else:
                self.best_bin = index
                self.best_bin_mu = bin_sample_mean


        def compute_CI(self, p):
            self.CI = [p, norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(1 - p),
                       norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(p)]
            return self.CI



        def get_prior_str(self):

            if self.tau[-2] > 0:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, {self.tau[-2] ** (-1):.2f})"
            else:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, \\infty)"

        def get_mean_str(self):
            return f"Posterior for mu ({self.band_index}): N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_mean_posterior_str(self):
            return f"N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_band_variance_est(self):
            return f"{self.est_band_var[-1]:.2f}"

        def get_variance_posterior_str(self):
            return f"Inx-X^2({self.df[-1]}, {self.eta[-1]:.2f})"


        def plot_posterior_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            ax.plot(x, self.nu[1:], label=f"B({self.band_index}) Posterior Mean")
            CI = 2 * np.power(self.tau, np.ones(len(
                self.tau)) * -1 / 2)  # Compute the posterior standard deviation and take 2 times this value as the 95% credible interval
            upper_CI = self.nu + CI
            lower_CI = self.nu - CI
            ax.fill_between(x, lower_CI[1:], upper_CI[1:], alpha=0.1, label="95% Credible Interval")

            # ax.plot(x, upper_CI[1:])
            # ax.plot(x, lower_CI[1:])
            ax.legend()

        def plot_posterior_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            var = np.power(self.tau, np.ones(len(self.tau)) * -1)
            ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
            ax.legend()

        def plot_band_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.nu))
            var = np.power(self.sigma_i_est, np.ones(len(self.sigma_i_est)) * -1)
            ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
            ax.legend()

        # <editor-fold desc="Unit Tests">
        # Unit Tests
        def sample_posterior_unit_test(self, k=10000):
            samples = self.sample_posterior(k)
            sample_mean = np.mean(samples)
            sample_scale = np.std(samples)
            print(f"The sample scale for the {k} samples of the posterior is $\hat\sigma$ = {sample_scale}")
        # </editor-fold>

    class Bin:
        # Store and update the belief for the band parameter (only for mean)
        def __init__(self, band, bin_index):
            self.band = band
            self.index = bin_index
            self.reward_variance = band.reward_variance
            self.true_band_var = band.true_band_var
            self.sample_mean = None
            self.ss = 0
            self.sample_var = None
            self.nu = None
            self.tau = 1 / self.reward_variance
            self.observations = []

        def add_observations(self, observations):
            self.observations.extend(observations)
            mu_ij = self.compute_mean(observations)
            if len(self.observations) == 1:
                self.band.obs_bins.append(self.index)
            self.band.update_best_bin(self.index, mu_ij)

        def compute_mean(self, observation):
            if self.sample_mean is None:  # first sample of this bin
                self.sample_mean = observation
            else:  # recursive mean calcluation
                n = len(self.observations)
                self.sample_mean = 1 / n * ((n - 1) * self.sample_mean + observation)
            return self.sample_mean

        def compute_posterior_parameters(self, mu_i):

            nb_obs = len(self.observations)
            if nb_obs > 0:  # If we have no observations,
                s_ij = self.reward_variance / nb_obs
                obs_mean = np.mean(self.observations)
                precision = 1 / s_ij + 1 / self.band.true_band_var
                numerator = 1 / s_ij * obs_mean + 1 / (self.band.true_band_var) * mu_i
                posterior_mean = numerator / precision
            else:
                precision = 1 / self.band.true_band_var
                posterior_mean = mu_i

            return posterior_mean, precision


class UCBEstimator(object):

    def __init__(self, environment, Settings):
        # If delta is None, use Algorithm 6 in Bandit Algorithms (Lattimore,2020)
        self.nbBands = environment.nbBands
        self.nbBins = environment.nbBins
        self.Type = Settings["UCB_Type"]
        self.delta = Settings["UCB_delta"]
        self.index = np.ones([self.nbBins*self.nbBands])*np.infty
        self.mu_hat = np.ones([self.nbBins*self.nbBands])
        self.counts = np.zeros([self.nbBins*self.nbBands])
        self.sums = np.zeros([self.nbBins*self.nbBands])
        self.rng = environment.rng


    def update(self, time, band_index, bin_index, reward):
        if bin_index is not None:
            arm_index = get_arm_index(band_index, bin_index, self.nbBands, self.nbBins)
        if time < self.nbBands:
            return
        self.counts[arm_index] +=1
        self.sums[arm_index] += reward
        self.mu_hat[arm_index] = self.sums[arm_index]/self.counts[arm_index]
        if self.Type == "classic":
            self.index[arm_index] = self.mu_hat[arm_index] + np.sqrt(2*np.log(time)/self.counts[arm_index])
        elif self.Type == "anytime":
            self.index[arm_index] = self.mu_hat[arm_index] + np.sqrt(
                2 * np.log(1 + time * np.log(time) ** 2) / self.counts[arm_index])
        elif self.Type == "gaussian":
            self.index[arm_index] = self.mu_hat[arm_index] + np.sqrt(2*np.log(time)/self.counts[arm_index])

        elif self.Type == "delta":
            self.index[arm_index] = self.mu_hat[arm_index] + np.sqrt(2*np.log(1/self.delta)/self.counts[arm_index])


        return

    def getMaxIndex(self):
        max_index = np.argmax(self.index)
        max_value = self.index[max_index]
        max_bb_index = get_bb_index(max_index, self.nbBands, self.nbBins)
        return (max_bb_index, max_index, max_value)


    def get_latest_params(self):
        return "NOT IMPLEMENTED"





# class UnknownVarianceEstimator2(object):
#     # Known Variance NormalNormal Estimator with uninformative band prior
#     def __init__(self, environment, sigmas_i, sigma_r, bands_priors=None, bins_priors=None, percentile=0.95):
#         self.nbBands = environment.nbBands
#         self.nbBins = environment.nbBins
#         self.bands = []
#         self.history = [[]]
#         self.sigmas_i = sigmas_i
#         self.sigma_r = sigma_r  # Known bin reward variance
#         self.id = None
#         self.latest_params = np.zeros([5, self.nbBands])
#         self.percentile = percentile
#         # Initialize Bands (and bins)
#         for b in range(self.nbBands):
#             self.bands.append(self.Band(self, b, bands_priors, bins_priors))
#
#     def update(self, t, band_index, bin_index, samples):
#         band = self.bands[band_index]
#         bin = band.bins[bin_index]
#         bin.add_observations(samples)  # Updating Bin Samples before band samples for non-sequential updates
#         sigma_i = band.update_sigma_i()
#         nu, tau_i = band.update_batch(t, bin_index, samples)
#         band.compute_CI(self.percentile)
#         self.latest_params[:, band_index] = np.array([[nu, tau_i ** (-1), band.CI[1], band.CI[2], sigma_i]])
#         return self.latest_params
#         # (old_nu, old_tau, new_nu, new_tau) = band.update_seq(t ,bin_index, samples, debug)
#
#     def sample_band_instances(self):
#         mu_i_samples = np.zeros(self.nbBands)
#         for k in range(self.nbBands):
#             mu_i_samples[k] = self.bands[k].sample_posterior()
#         return mu_i_samples, self.sigmas_i
#
#     def sample_bin_instances(self, band_index, mu_i):
#         band = self.bands[band_index]
#         nu_ij_samples = np.zeros(self.nbBins)
#         tau_ij_samples = np.zeros(self.nbBins)
#         for k in range(self.nbBins):
#             (nu_ij_samples[k], tau_ij_samples[k]) = band.bins[k].compute_posterior_parameters(mu_i)
#         return nu_ij_samples, tau_ij_samples
#
#     def print_band_parameter_estimates(self, band_index):
#         for b in band_index:
#             print(self.bands[b].get_posterior_str())
#
#     class Band:
#         # Store and update the belief for the band parameters
#         '''
#         The band parameters are nu and tau_i, and we store it after each update in list fashion
#         '''
#
#         def __init__(self, estimator, band_index, band_priors=None, bin_prior=None):
#             self.estimator = estimator
#             self.band_index = band_index
#             self.nbBins = estimator.nbBins
#             self.sigma_i = estimator.sigmas_i[band_index]
#             self.sigma_i_est = [np.infty]
#             self.true_sigma_i = estimator.sigmas_i[band_index]
#
#             self.sigma_r = estimator.sigma_r
#             # self.uninformative = True  # We start with a non-informative prior over the mean
#             self.pulls = 0
#             self.history = [[]]
#             self.bins = []
#             self.obs_bins = []
#
#             self.nu = [0]
#             self.tau_i = [0]
#
#             if band_priors is not None:  # We have priors for the band parameters
#                 self.nu[0] = band_priors["nu"]
#                 self.tau_i[0] = band_priors["tau_i"]
#
#             for b in range(self.nbBins):
#                 self.bins.append(UnknownVarianceEstimator.Bin(self, b, bin_prior))
#
#         def update_sigma_i(self):
#             if len(self.obs_bins) < 2:
#                 self.sigma_i_est = [np.infty]
#             else:
#                 sample_means = [self.bins[bin].sample_mean for bin in self.obs_bins]
#                 self.sigma_i_est.append(np.std(sample_means))
#             print(f"Estimated Band std = {self.sigma_i_est[-1]}")
#             return self.sigma_i_est[-1]
#
#         def update_batch(self, bin_index, t, samples):
#             tau_i = 0
#             numerator = 0
#             for bin in self.obs_bins:
#                 bin = self.bins[bin]
#                 obs = bin.observations
#                 s_ij = bin.sigma_r ** 2 / len(obs)
#                 tau_i += 1 / (s_ij + self.sigma_i ** 2)
#                 numerator += np.mean(obs) / (s_ij + self.sigma_i ** 2)
#             nu = numerator / tau_i
#             self.nu.append(nu)
#             self.tau_i.append(tau_i)
#             self.history.append([[t], [bin_index], [samples]])
#             self.pulls += 1
#             if DS["update_batch"]:
#                 print(self.get_prior_str())
#                 print(self.get_posterior_str())
#             return nu, tau_i
#
#         def compute_CI(self, p):
#             self.CI = [p, norm(loc=self.nu[-1], scale=self.tau_i[-1] ** (-1 / 2)).ppf(1 - p),
#                        norm(loc=self.nu[-1], scale=self.tau_i[-1] ** (-1 / 2)).ppf(p)]
#             return self.CI
#
#         def sample_posterior(self, k=1):
#             return norm.rvs(loc=self.nu[-1], scale=max(self.tau_i[-1], .000001) ** (-1 / 2), size=k)
#
#         def get_prior_str(self):
#
#             if self.tau_i[-2] > 0:
#                 return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, {self.tau_i[-2] ** (-1):.2f})"
#             else:
#                 return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, \\infty)"
#
#         def get_posterior_str(self):
#             return f"Posterior for mu ({self.band_index}): N({self.nu[-1]:.2f}, {self.tau_i[-1] ** (-1):.2f})"
#
#         def posterior_str(self):
#             return f"N({self.nu[-1]:.2f}, {self.tau_i[-1] ** (-1):.2f})"
#
#         def plot_posterior_history(self, ax=None):
#             if ax is None:
#                 fig, ax = plt.subplots()
#             x = np.arange(1, len(self.nu))
#             ax.plot(x, self.nu[1:], label=f"B({self.band_index}) Posterior Mean")
#             CI = 2 * np.power(self.tau_i, np.ones(len(
#                 self.tau_i)) * -1 / 2)  # Compute the posterior standard deviation and take 2 times this value as the 95% credible interval
#             upper_CI = self.nu + CI
#             lower_CI = self.nu - CI
#             ax.fill_between(x, lower_CI[1:], upper_CI[1:], alpha=0.1, label="95% Credible Interval")
#
#             # ax.plot(x, upper_CI[1:])
#             # ax.plot(x, lower_CI[1:])
#             ax.legend()
#
#         def plot_posterior_variance_history(self, ax=None):
#             if ax is None:
#                 fig, ax = plt.subplots()
#             x = np.arange(1, len(self.nu))
#             var = np.power(self.tau_i, np.ones(len(self.tau_i)) * -1)
#             ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
#             ax.legend()
#
#         def plot_band_variance_history(self, ax=None):
#             if ax is None:
#                 fig, ax = plt.subplots()
#             x = np.arange(1, len(self.nu))
#             var = np.power(self.sigma_i_est, np.ones(len(self.sigma_i_est)) * -1)
#             ax.plot(x, var[1:], label=f"B({self.band_index}) Posterior Variance ")
#             ax.legend()
#
#         # <editor-fold desc="Unit Tests">
#         # Unit Tests
#         def sample_posterior_unit_test(self, k=10000):
#             samples = self.sample_posterior(k)
#             sample_mean = np.mean(samples)
#             sample_scale = np.std(samples)
#             print(f"The sample scale for the {k} samples of the posterior is $\hat\sigma$ = {sample_scale}")
#         # </editor-fold>
#
#     class Bin:
#         # Store and update the belief for the band parameter (only for mean)
#         def __init__(self, band, bin_index, bin_prior=None):
#             self.band = band
#             self.index = bin_index
#             self.sigma_r = band.sigma_r
#             self.sigma_i = band.sigma_i
#             self.sample_mean = None
#             self.ss = 0
#             self.sample_var = None
#             self.nu = None
#             self.tau_i = 1 / self.sigma_i ** 2
#             self.observations = []
#             if bin_prior is not None:
#                 self.nu = bin_prior["nu"]
#                 self.rho = bin_prior["rho"]
#
#         def add_observations(self, observations):
#             self.observations.extend(observations)
#             self.compute_mean(observations)
#             if len(self.observations) == 1:
#                 self.band.obs_bins.append(self.index)
#
#         def compute_mean(self, observation):
#             if self.sample_mean is None:  # first sample of this bin
#                 self.sample_mean = observation
#             else:  # recursive mean calcluation
#                 n = len(self.observations)
#                 self.sample_mean = 1 / n * ((n - 1) * self.sample_mean + observation)
#
#         def compute_posterior_parameters(self, mu_i):
#
#             nb_obs = len(self.observations)
#             if nb_obs > 0:  # If we have no observations,
#                 s_ij = self.sigma_r ** 2 / nb_obs
#                 obs_mean = np.mean(self.observations)
#             else:
#                 s_ij = np.infty
#                 obs_mean = 0
#             precision = 1 / s_ij + 1 / self.sigma_i ** 2
#             numerator = 1 / s_ij * obs_mean + 1 / (self.sigma_i ** 2) * mu_i
#             posterior_mean = numerator / precision
#             return posterior_mean, precision