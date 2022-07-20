import numpy as np
from scipy.stats import norm, invgamma
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from config import DEBUGSETTINGS as DS

class KnownVarianceEstimator(object):
    # Known Variance NormalNormal Estimator with uninformative band prior
    def __init__(self, environment, sigmas_i, sigma_r, bands_priors=None, bins_priors=None, percentile=0.95):
        self.nbBands = environment.nbBands
        self.nbBins = environment.nbBins
        self.bands = []
        self.history = [[]]
        self.sigmas_i = sigmas_i  # Known band gen distribution variances
        self.sigma_r = sigma_r  # Known bin reward variance
        self.id = None
        self.latest_params = np.zeros([4, self.nbBands])
        self.percentile = percentile
        # Initialize Bands (and bins)
        for b in range(self.nbBands):
            self.bands.append(self.Band(self, b, bands_priors, bins_priors))

    def update(self, t, band_index, bin_index, samples):
        band = self.bands[band_index]
        bin = band.bins[bin_index]
        bin.add_observations(samples)  # Updating Bin Samples before band samples for non-sequential updates
        nu, tau = band.update_batch(t, bin_index, samples)
        band.compute_CI(self.percentile)
        self.latest_params[:, band_index] = np.array([[nu, tau ** (-1), band.CI[1], band.CI[2]]])
        return self.latest_params
        # (old_nu, old_tau, new_nu, new_tau) = band.update_seq(t ,bin_index, samples, debug)

    def sample_band_instances(self):
        mu_i_samples = np.zeros(self.nbBands)
        for k in range(self.nbBands):
            mu_i_samples[k] = self.bands[k].sample_posterior()
        return mu_i_samples, self.sigmas_i

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

        def __init__(self, estimator, band_index, band_priors=None, bin_prior=None):
            self.estimator = estimator
            self.band_index = band_index
            self.nbBins = estimator.nbBins
            self.sigma_i = estimator.sigmas_i[band_index]
            self.sigma_r = estimator.sigma_r
            # self.uninformative = True  # We start with a non-informative prior over the mean
            self.pulls = 0
            self.history = [[]]
            self.bins = []
            self.obs_bins = []

            self.nu = [0]
            self.tau = [0]

            if band_priors is not None:  # We have priors for the band parameters
                self.nu[0] = band_priors["nu"]
                self.tau[0] = band_priors["tau_i"]

            for b in range(self.nbBins):
                self.bins.append(KnownVarianceEstimator.Bin(self, b, bin_prior))

        def update_batch(self, bin_index, t, samples):
            tau = 0
            numerator = 0
            for bin in self.obs_bins:
                bin = self.bins[bin]
                obs = bin.observations
                s_ij = bin.sigma_r ** 2 / len(obs)
                tau += 1 / (s_ij + self.sigma_i ** 2)
                numerator += np.mean(obs) / (s_ij + self.sigma_i ** 2)
            nu = numerator / tau
            self.nu.append(nu)
            self.tau.append(tau)
            self.history.append([[t], [bin_index], [samples]])
            self.pulls += 1
            if DS["update_batch"]:
                print(self.get_prior_str())
                print(self.get_posterior_str())
            return nu, tau

        def update_seq(self, t, bin_index, samples, debug=False):
            sample_mean = np.mean(samples)
            nb_samples = np.shape(samples)[0]
            precision_factor = (self.sigma_i ** 2 + self.sigma_r ** 2 / nb_samples) ** (-1)
            old_tau = self.tau[-1]
            old_nu = self.nu[-1]
            new_tau = old_tau + precision_factor
            new_nu = 1 / new_tau * (old_tau * old_nu + precision_factor * sample_mean)
            self.tau.append(new_tau)
            self.nu.append(new_nu)
            self.history.append([[t], [bin_index], [samples]])
            self.pulls += nb_samples
            if debug:
                print(self.get_prior_str())
                print(self.get_posterior_str())
            return (old_nu, old_tau, new_nu, new_tau)

        def compute_CI(self, p):
            self.CI = [p, norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(1 - p),
                       norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(p)]
            return self.CI

        def sample_posterior(self, k=1):
            return norm.rvs(loc=self.nu[-1], scale=max(self.tau[-1], .000001) ** (-1 / 2), size=k,
                            random_state=self.rng)

        def get_prior_str(self):

            if self.tau[-2] > 0:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, {self.tau[-2] ** (-1):.2f})"
            else:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, \\infty)"

        def get_posterior_str(self):
            return f"Posterior for mu ({self.band_index}): N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def posterior_str(self):
            return f"N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

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
        def __init__(self, band, bin_index, bin_prior=None):
            self.band = band
            self.index = bin_index
            self.sigma_r = band.sigma_r
            self.sigma_i = band.sigma_i
            self.nu = None
            self.tau = 1 / self.sigma_i ** 2
            self.observations = []
            if bin_prior is not None:
                self.nu = bin_prior["nu"]
                self.rho = bin_prior["rho"]

        def add_observations(self, observations):
            self.observations.extend(observations)
            if len(self.observations) == 1:
                self.band.obs_bins.append(self.index)

        def compute_posterior_parameters(self, mu_i):

            nb_obs = len(self.observations)
            if nb_obs > 0:  # If we have no observations,
                s_ij = self.sigma_r ** 2 / nb_obs
                obs_mean = np.mean(self.observations)
            else:
                s_ij = np.infty
                obs_mean = 0
            precision = 1 / s_ij + 1 / self.sigma_i ** 2
            numerator = 1 / s_ij * obs_mean + 1 / (self.sigma_i ** 2) * mu_i
            posterior_mean = numerator / precision
            return posterior_mean, precision


class UnknownVarianceEstimator(object):
    # Factors in estimates of the band variance into the posterior distribution of band means
    def __init__(self, environment, percentile=0.95, var_method=0):
        self.nbBands = environment.nbBands
        self.nbBins = environment.nbBins
        self.bands = []
        self.history = {}
        self.true_band_vars = environment.band_variances  # True band variance (only used for debugging)
        self.reward_variance = environment.reward_variance  # Known bin reward variance
        self.var_method = 0
        self.id = None
        self.latest_params = np.zeros([5, self.nbBands])
        self.percentile = percentile  # Confidence Interval Percentile
        # Initialize Bands (and bins)
        for b in range(self.nbBands):
            self.bands.append(self.Band(self, b))

    def update(self, t, band_index, bin_index, samples):
        band = self.bands[band_index]
        bin = band.bins[bin_index]
        bin.add_observations(samples)  # Updating Bin Samples before band samples for non-sequential updates
        band.add_observations(t, bin_index, samples)
        sigma_i = band.update_band_var_est()
        nu, tau = band.update_theta_posterior(t, bin_index, samples)
        band.compute_CI(self.percentile)
        self.latest_params[:, band_index] = np.array([[nu, tau ** (-1), band.CI[1], band.CI[2], sigma_i]])
        return self.latest_params
        # (old_nu, old_tau, new_nu, new_tau) = band.update_seq(t ,bin_index, samples, debug)

    def sample_band_instances(self):
        mu_i_samples = np.zeros(self.nbBands)
        var_i_samples = np.zeros(self.nbBands)
        for k in range(self.nbBands):
            mu_i_samples[k] = self.bands[k].sample_posterior()
            var_i_samples[k] = self.bands[k].est_band_var[-1]
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

            self.nu = [0]
            self.tau = [0]

            for b in range(self.nbBins):
                self.bins.append(UnknownVarianceEstimator.Bin(self, b))

        def add_observations(self, time, bin, samples):
            self.history["Time"].append(time)
            self.history["Bin"].append(bin)
            self.history["Reward"].append(samples)

        def update_band_var_est(self):
            def compute_weighted_var():
                # First compute weighted mean
                sample_means = [self.bins[bin].sample_mean for bin in self.obs_bins]
                samples = [len(self.bins[bin].observations) for bin in self.obs_bins]
                temp = np.product([sample_means, samples])
                temp2 = np.mean(temp / sum(samples))
                var = 1 / len(samples) * np.sum(sample_means - temp2) ** 2
                return var

            if self.estimator.var_method == 0:
                if len(self.obs_bins) < 2:
                    self.est_band_var = [np.infty]
                else:
                    sample_means = [self.bins[bin].sample_mean for bin in self.obs_bins]
                    self.est_band_var.append(np.var(sample_means))
                if DS["update_batch"]:
                    print(f"Estimated Band Variance = {self.est_band_var[-1]}")
                return self.est_band_var[-1]
            elif self.estimator.var_method == 1:
                # Weight by number of samples per bin
                if len(self.obs_bins) < 2:
                    self.est_band_var = [np.infty]
                else:
                    self.est_band_var.append(compute_weighted_var())
                if DS["update_batch"]:
                    print(f"Estimated Band Variance = {self.est_band_var[-1]}")
                return self.est_band_var[-1]

        def update_theta_posterior(self, bin_index, t, samples):
            tau = 0
            numerator = 0
            for bin in self.obs_bins:

                bin = self.bins[bin]
                obs = bin.observations
                s_ij = bin.reward_variance / len(obs)
                if self.est_band_var[-1] < 1000:
                    tau += 1 / (s_ij + self.est_band_var[-1])
                else:
                    tau += 0.00001
                numerator += np.mean(obs) / (s_ij + self.est_band_var[-1])
            nu = numerator / tau
            if nu > 1e4:
                nu = 0
                tau = 10
            self.nu.append(nu)
            self.tau.append(tau)
            # self.history.append([[t], [bin_index], [samples]])
            self.pulls += 1
            if DS["update_batch"]:
                print(self.get_prior_str())
                print(self.get_posterior_str())
            return nu, tau

        def compute_CI(self, p):
            self.CI = [p, norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(1 - p),
                       norm(loc=self.nu[-1], scale=self.tau[-1] ** (-1 / 2)).ppf(p)]
            return self.CI

        def sample_posterior(self, k=1):
            return norm.rvs(loc=self.nu[-1], scale=max(self.tau[-1], .000001) ** (-1 / 2), size=k,
                            random_state=self.rng)

        def get_prior_str(self):

            if self.tau[-2] > 0:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, {self.tau[-2] ** (-1):.2f})"
            else:
                return f"Prior for mu ({self.band_index}): N({self.nu[-2]:.2f}, \\infty)"

        def get_posterior_str(self):
            return f"Posterior for mu ({self.band_index}): N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def posterior_str(self):
            return f"N({self.nu[-1]:.2f}, {self.tau[-1] ** (-1):.2f})"

        def get_band_variance_est(self):
            return f"{self.est_band_var[-1]:.2f}"

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
                precision = 1 / s_ij + 1 / self.band.est_band_var[-1]
                numerator = 1 / s_ij * obs_mean + 1 / (self.band.est_band_var[-1]) * mu_i
                posterior_mean = numerator / precision
            else:
                precision = 1 / self.band.est_band_var[-1]
                posterior_mean = mu_i

            return posterior_mean, precision