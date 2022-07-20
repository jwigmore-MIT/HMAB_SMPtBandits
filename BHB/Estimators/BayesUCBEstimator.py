import numpy as np
from BHB.Estimators.BaseEstimator import BaseEstimator
from BHB.helper import *
from scipy.stats import norm



class BayesUCBEstimator(BaseEstimator):

    def __init__(self, environment, settings, est_priors = None, confidence_level = .95):
        self.env = environment
        self.type = "UCB"
        self.horizon = settings["Horizon"]
        self.nbClusters = environment.nbClusters
        self.nbArms = environment.nbArms
        self.confidence_level = confidence_level
        self.rng = environment.rng
        self.arm_beliefs = []
        self.UCB_indices = []
        self.prior_means = []
        self.prior_variances = []
        self.counts = np.zeros([self.nbArms * self.nbClusters])
        for n in range(self.nbClusters):
            if est_priors is None:
                self.prior_means.append(self.env.hyperprior_means[n])
                self.prior_variances.append(self.env.hyperprior_variances[n]+self.env.cluster_variances[n])
            else: # Allow for misspecification or uninformative priors
                self.prior_means.append(est_priors["hyperprior_means"][n])
                self.prior_variances.append(est_priors["hyperprior_variances"][n]+self.env.cluster_variances[n])
            for m in range(self.nbArms):
                self.arm_beliefs.append(self.ArmBelief(self, n, m, self.prior_means[n], self.prior_variances[n]))
                self.UCB_indices.append(np.infty)

    def update(self, time, cluster_index, arm_index, reward):
        arm_total_index = get_1D_arm_index(self, cluster_index, arm_index)
        new_UCB_index = self.arm_beliefs[arm_total_index].update(reward)
        self.counts[arm_total_index] += 1
        self.UCB_indices[arm_total_index] = new_UCB_index

    def compute_indices(self, time):
        for arm_num in range(self.nbArms * self.nbClusters):
            self.UCB_indices[arm_num] = self.arm_beliefs[arm_num].get_index(time)


    def getMaxIndex(self):
        max_index = np.argmax(self.UCB_indices)
        max_value = self.UCB_indices[max_index]
        max_cc_index = get_2D_arm_index(self, max_index)
        return (max_cc_index, max_index, max_value)

    def get_latest_params(self):
        return "NOT IMPLEMENTED"



    class ArmBelief:

        def __init__(self, estimator, cluster_index, arm_index, prior_mean, prior_variance, ):
            self.estimator = estimator
            self.cluster_index = cluster_index
            self.arm_index = arm_index
            self.total_index = get_1D_arm_index(estimator, cluster_index, arm_index)
            self.prior_mean = prior_mean
            self.prior_variance = prior_variance
            self.reward_variance = estimator.env.reward_variance
            self.observations = []
            self.v_ij = [prior_variance]
            self.u_ij = [prior_mean]


        def update(self, reward):
            self.observations.append(reward)
            K = len(self.observations)
            sum_rewards = np.sum(self.observations)
            v_ij = 1/(1/self.prior_variance+K/self.reward_variance)
            u_ij = v_ij * (self.prior_mean/self.prior_variance + sum_rewards/self.reward_variance)
            self.v_ij.append(v_ij)
            self.u_ij.append(u_ij)

        def get_index(self, time):
            self.index = norm.ppf(1-1/((time+1)*self.estimator.confidence_level*np.log(self.estimator.horizon)) , loc = self.u_ij[-1], scale = np.sqrt(self.v_ij[-1]))
            return self.index
