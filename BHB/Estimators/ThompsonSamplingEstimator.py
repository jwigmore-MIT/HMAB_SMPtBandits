from typing import Union
from BHB.Estimators.BaseEstimator import BaseEstimator
import numpy as np
from scipy.stats import norm, invgamma
import matplotlib
import matplotlib.pyplot as plt
from BHB.helper import *

from copy import deepcopy
# from analysistools import pr_max_X, sim_pr_max
from BHB.config import DEBUGSETTINGS as DS
np.seterr(all = "raise")


class ThompsonSamplingEstimator(BaseEstimator):
    # Sample as known variance estimator but ignores the theta_i samples when
    # sampling from the arm posteriors i.e. samples from independent marginal posteriors for
    # all arm parameters
    # Not the most efficient implementation, but easy modification from previous code
    def __init__(self, environment, est_priors = None):
        self.env = environment
        self.nbClusters = environment.nbClusters
        self.nbArms = environment.nbArms
        if est_priors is None:
            self.hyperprior_means = self.env.hyperprior_means
            self.hyperprior_variances = self.env.hyperprior_variances
        elif est_priors == "Uninformative":
            self.hyperprior_means = np.zeros(self.nbClusters)
            self.hyperprior_variances = np.ones(self.nbClusters)*10e7
        else:
            self.hyperprior_means = est_priors["hyperprior_means"]
            self.hyperprior_variances = est_priors["hyperprior_variances"]
        self.cluster_beliefs = []
        self.history = {}
        self.cluster_variances = self.env.cluster_variances  #
        self.reward_variance = self.env.reward_variance  # Known arm reward variance
        self.id = None
        self.parameter_strings = ["v_i", "u_i", "theta_samples"]
        self.rng = environment.rng
        # Initialize Clusters (and arms)
        for b in range(self.nbClusters):
            self.cluster_beliefs.append(
                self.ClusterBelief(self, b, self.hyperprior_means[b], self.hyperprior_variances[b]))


    def update(self, t:int, cluster_index: int, arm_index: int, rewards: Union[int,list]):
        """
        Updates the history from a(t) = (cluster_index, arm_index) with observed reward(s)
        :param t: round of observation
        :param cluster_index: chosen cluster
        :param arm_index: chosen arm
        :param rewards: observed reward(s)
        :return:
        """
        self.cluster_beliefs[cluster_index].update(t, arm_index, rewards)



    def get_latest_params(self):
        latest_params = np.zeros([3, self.nbClusters])
        for b in range(self.nbClusters):
            latest_params[0,b] = self.cluster_beliefs[b].tau_i[-1] ** (-1)
            latest_params[1,b] = self.cluster_beliefs[b].u_i[-1]
            latest_params[2,b] = self.cluster_beliefs[b].theta_i_samples[-1]
        return latest_params

    def full_hier_sampling(self):
        '''
        Full Hierarchical Sampling
        :return:
        '''

        theta_i_samples = self.sample_cluster_instances()
        theta_ij_samples = np.zeros((self.nbClusters, self.nbArms))
        for k in range(self.nbClusters):
            theta_ij_samples[k,:] = self.cluster_beliefs[k].sample_all_arm_beliefs(theta_i_samples[k])
        return np.reshape(theta_ij_samples, (self.nbArms*self.nbClusters,1))

    def sample_cluster_instances(self):
        theta_i_samples = np.zeros(self.nbClusters)
        for k in range(self.nbClusters):
            theta_i_samples[k] = self.cluster_beliefs[k].sample_belief() # Sample from the conditional posterior for the cluster mean
        return theta_i_samples

    def sample_best_arms(self, mu_i_samples: float, sigma_i_samples: float):
        """

        :param mu_i_samples: samples of mu_i for each cluster from their respective posteriors
        :param sigma_i_samples: true cluster variance (but this is retrieved in the arm.compute_posterior_parameter() so
               unused
        :return: [list, list, list]: lists of samples from posterior, posterior mean parameters, and
                                     posterior precision parameters
        """
        u_ij_samples = np.zeros(self.nbClusters)
        tau_ij_samples = np.zeros(self.nbClusters)
        theta_ij_samples = np.zeros(self.nbArms)

        for k in range(self.nbClusters):
            best_arm_index = self.clusters[k].best_arm # index of best arm
            u_ij_samples[k], tau_ij_samples[k] = self.clusters[k].arms[best_arm_index].compute_posterior_parameters(mu_i_samples[k])
            theta_ij_samples[k] = norm.rvs(loc = u_ij_samples[k], scale = tau_ij_samples[k]**(-1/2), random_state= self.rng)

        return theta_ij_samples, u_ij_samples, tau_ij_samples

    def sample_arm_instances(self, cluster_index, theta_i):
        cluster = self.cluster_beliefs[cluster_index]
        theta_ij_samples = np.zeros(self.nbArms)
        for k in range(self.nbArms):
            theta_ij_samples[k]= cluster.arm_beliefs[k].sample_arm_belief(theta_i)
        return theta_ij_samples


    def get_cluster_posterior_strs(self, cluster_index = "all", printit = False):
        if cluster_index == 'all':
            cluster_index = range(self.nbClusters)
        post_strs = []
        for b in cluster_index:
            post_strs.append(self.cluster_beliefs[b].get_mean_str())
            if printit: print(post_strs[-1])
        return post_strs




    class ClusterBelief:
        # Store and update the belief for the cluster parameters
        '''
        The cluster parameters are nu and tau_i, and we store it after each update in list fashion
        '''

        def __init__(self, estimator, cluster_index, hyperprior_mean, hyperprior_variance):
            self.estimator = estimator
            self.cluster_index = cluster_index
            self.nbArms = estimator.nbArms
            self.cluster_var = estimator.cluster_variances[cluster_index] # sigma_i for this cluster
            self.reward_variance = estimator.reward_variance # sigma_ij for all arms in this cluster
            self.hyperprior_mean = self.estimator.hyperprior_means[cluster_index] # kappa_i
            self.hyperprior_variance = self.estimator.hyperprior_variances[cluster_index] # gamma_i
            self.pulls = 0
            self.history = {
                "Time": [],
                "ArmBelief": [],
                "Reward": []
            }
            self.arm_beliefs = []
            self.obs_arms = [] #Observed arms - should be all
            self.u_i = [hyperprior_mean]  # Location parameter for the posterior distribution on the cluster mean
            self.tau_i = [hyperprior_variance**(-1)]  # Scale (Precision_ parameter for the posterior distribution on the cluster mean
            self.theta_i_samples = [np.nan] #Samples taken from belief distribution
            self.best_arm = None
            self.best_arm_mu = -np.infty


            for b in range(self.nbArms):
                self.arm_beliefs.append(ThompsonSamplingEstimator.ArmBelief(self, b))

        def update(self, t: int, arm_index: int, rewards: Union[float, list]):
            """
                Updates the history from a(t) = (arm_index) with observed reward(s)
                :param t: round of observation
                :param arm_index: chosen arm
                :param rewards: observed reward(s)
                :return:
                """
            self.arm_beliefs[arm_index].add_observations(rewards)
            self.add_observations(t, arm_index, rewards)
            self.compute_posterior_belief()


        def add_observations(self, time, arm, samples):
            '''Updates the history for use of the results'''
            self.history["Time"].append(time)
            self.history["ArmBelief"].append(arm)
            self.history["Reward"].append(samples)

        def compute_posterior_belief(self):
            tau_i = 0
            numerator = 0
            for arm_index in self.obs_arms:
                arm = self.arm_beliefs[arm_index]
                obs = arm.observations
                s_ij = arm.reward_variance / len(obs)
                tau_i += 1 / (s_ij + self.cluster_var ** 2)
                numerator += np.mean(obs) / (s_ij + self.cluster_var ** 2)

            u_i = numerator / tau_i
            self.u_i.append(u_i)
            self.tau_i.append(tau_i)

        def sample_belief(self):
            theta_i = norm.rvs(loc = self.u_i[-1], scale = self.tau_i[-1]**(-1/2), random_state= self.estimator.rng)
            self.theta_i_samples.append(theta_i)
            return theta_i

        def sample_all_arm_beliefs(self, theta_i):
            theta_ij = np.zeros((1,self.nbArms))
            for k in range(self.nbArms):
                theta_ij[0,k] = self.arm_beliefs[k].sample_arm_belief(theta_i)
            return theta_ij

        # def update_best_arm(self, index, arm_sample_mean):
        #     """
        #     Updates recorded best arm if computed arm_sample_mean is the max seen so far
        #     Gets called from child arm.add_observation
        #
        #     :param index:
        #     :param arm_sample_mean:
        #     :return:
        #     """
        #     if arm_sample_mean < self.best_arm_mu:
        #         return
        #     else:
        #         self.best_arm = index
        #         self.best_arm_mu = arm_sample_mean
        #
        #
        # def compute_CI(self, p):
        #     self.CI = [p, norm(loc=self.nu[-1], scale=self.tau_i[-1] ** (-1 / 2)).ppf(1 - p),
        #                norm(loc=self.nu[-1], scale=self.tau_i[-1] ** (-1 / 2)).ppf(p)]
        #     return self.CI

        def get_prior_str(self):

            if self.tau_i[-2] > 0:
                return f"Prior for mu ({self.cluster_index}): N({self.u_i[-2]:.2f}, {self.tau_i[-2] ** (-1):.2f})"
            else:
                return f"Prior for mu ({self.cluster_index}): N({self.u_i[-2]:.2f}, \\infty)"

        def get_mean_str(self):
            return f"Posterior for mu ({self.cluster_index}): N({self.u_i[-1]:.2f}, {self.tau_i[-1] ** (-1):.2f})"

        def get_cluster_belief_str(self):
            return f"N({self.u_i[-1]:.2f}, {self.tau_i[-1] ** (-1):.2f})"

        def get_cluster_variance_est(self):
            return f"{self.est_cluster_var[-1]:.2f}"

        def get_variance_posterior_str(self):
            return f"Inx-X^2({self.df[-1]}, {self.eta[-1]:.2f})"


        def plot_posterior_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.u_i))
            ax.plot(x, self.u_i[1:], label=f"B({self.cluster_index}) Posterior Mean")
            CI = 2 * np.power(self.tau_i, np.ones(len(
                self.tau_i)) * -1 / 2)  # Compute the posterior standard deviation and take 2 times this value as the 95% credible interval
            upper_CI = self.u_i + CI
            lower_CI = self.u_i - CI
            ax.fill_between(x, lower_CI[1:], upper_CI[1:], alpha=0.1, label="95% Credible Interval")

            # ax.plot(x, upper_CI[1:])
            # ax.plot(x, lower_CI[1:])
            ax.legend()

        def plot_posterior_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.u_i))
            var = np.power(self.tau_i, np.ones(len(self.tau_i)) * -1)
            ax.plot(x, var[1:], label=f"B({self.cluster_index}) Posterior Variance ")
            ax.legend()

        def plot_cluster_variance_history(self, ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x = np.arange(1, len(self.u_i))
            var = np.power(self.sigma_i_est, np.ones(len(self.sigma_i_est)) * -1)
            ax.plot(x, var[1:], label=f"B({self.cluster_index}) Posterior Variance ")
            ax.legend()

        # <editor-fold desc="Unit Tests">
        # Unit Tests
        def sample_posterior_unit_test(self, k=10000):
            samples = self.sample_posterior(k)
            sample_mean = np.mean(samples)
            sample_scale = np.std(samples)
            print(f"The sample scale for the {k} samples of the posterior is $\hat\sigma$ = {sample_scale}")
        # </editor-fold>

    class ArmBelief:
        # Store and update the belief for the cluster parameter (only for mean)
        def __init__(self, cluster, arm_index):
            self.cluster = cluster
            self.index = arm_index
            self.prior_mean = cluster.hyperprior_mean
            self.reward_variance = cluster.reward_variance
            self.prior_variance = cluster.hyperprior_variance + cluster.cluster_var
            self.sample_mean = None
            self.sample_var = None
            self.u_ij = None
            self.observations = []

        def add_observations(self, observations):
            self.observations.extend(observations)
            sample_mean = self.compute_sample_mean(observations)
            if len(self.observations) == 1:
                self.cluster.obs_arms.append(self.index)

        def compute_sample_mean(self, observation):
            if self.sample_mean is None:  # first sample of this arm
                self.sample_mean = observation
            else:  # recursive mean calcluation
                n = len(self.observations)
                self.sample_mean = 1 / n * ((n - 1) * self.sample_mean + observation)
            return self.sample_mean

        def compute_belief_params(self, theta_i = None):
            # Modified to ignore the sampled cluster parameter a

            nb_obs = len(self.observations)
            if nb_obs > 0:  # If we have no observations,
                s_ij = self.reward_variance / nb_obs
                tau_ij = 1 / s_ij + 1 /self.prior_variance
                u_ij = 1/tau_ij * (self.prior_mean/ self.prior_variance+self.sample_mean/s_ij)
            else:
                tau_ij = 1 / self.prior_variance
                u_ij = self.prior_mean

            return u_ij, tau_ij

        def sample_arm_belief(self, theta_i):
            u_ij, tau_ij = self.compute_belief_params(theta_i)
            theta_ij_sample = norm.rvs(loc = u_ij, scale = tau_ij**(-1/2), random_state= self.cluster.estimator.rng)
            return theta_ij_sample