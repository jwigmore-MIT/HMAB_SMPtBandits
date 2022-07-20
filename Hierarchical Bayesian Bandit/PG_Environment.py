import numpy as np
from scipy.stats import norm, t
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

class PG_Environment(object):
    # Prior Generatived Environment


    def __init__(self, scenario):
        self.nbClusters = scenario["nbClusters"]      # Number of clusters
        self.nbArms = scenario["nbArms"]        # Number of arms
        self.hyperprior_means = scenario["hyperprior_mean"]
        self.hyperprior_variances = scenario["hyperprior_variances"]
        self.cluster_means = []
        self.cluster_variances = scenario["cluster_variance"] # Variance of cluster distributions
        self.reward_variance = scenario["reward_variance"]  #
        self.rng = scenario["rng"]
        self.clusters = []
        self.arms = []
        self.arm_means = []
        self.best_arms = []
        self.best_means = []
        self.init_clusters()

    def init_clusters(self):
        for b in range(self.nbClusters):
            self.cluster_means.append(norm.rvs(self.hyperprior_means[b], self.hyperprior_variances[b],
                                               random_state= self.rng))
            self.clusters.append(self.Cluster(self, b, self.nbArms, self.cluster_meanp[b], self.cluster_variances[b],
                                              self.reward_variance))
            self.arm_means.append(self.clusters[b].arm_means) # The initialization of the cluster
                                                              # will generate all mean arm reward values
        for arm_list in self.arm_means:
            self.best_arms.append(np.argmax(arm_list))
            self.best_means.append(np.max(arm_list))



    def draw_samples(self, cluster_index, arm_index, k=1):
        return self.clusters[cluster_index].arms[arm_index].sample(k)


    # <editor-fold desc="Plotting Methods">
    def plot_clusters_dists(self, fig = None, ax = None, gen = True, samp = True, threshold = None):
        if fig is None:
            fig, ax = plt.subplots()
        for cluster in self.clusters:
            cluster.plot_dists(fig, ax, gen, samp)
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
        for b in range(self.nbClusters):
            tails.append(self.clusters[b].compute_tail(threshold))
        self.tail_mass = tails

    class Cluster(object):

        def __init__(self, Environment, cluster_index, nbArms, cluster_mean, cluster_var, arm_params):
            self.Environment = Environment
            self.index = cluster_index
            self.nbArms = nbArms
            self.mu = cluster_mean
            self.var = cluster_var
            self.sigma = np.sqrt(self.var)
            self.arm_params = arm_params
            self.rv = norm(loc = self.mu, scale = self.sigma)
            self.arms = []
            self.init_arms()
            self.id = None

        def init_arms(self):
            arm_means = self.rv.rvs(size = self.nbArms, random_state = self.Environment.rng)
            for b in range(self.nbArms):
                self.arms.append(Environment.ArmBelief(self, b, arm_means[b], self.arm_params))
            self.sort_index = np.argsort(arm_means)
            self.sample_mu = np.mean(arm_means)
            self.sample_sigma = np.std(arm_means)
            self.arm_means = arm_means

        # <editor-fold desc="STR METHODS">
        def __str__(self):
            return f"""Cluster {self.index} ~ N({self.mu:.1f}, {self.sigma**2:.2f})/N({self.sample_mu:.1f}, {self.sample_sigma**2:.2f})"""

        def __genstr__(self):
            return f"""Cluster {self.index} (G) ~ N({self.mu:.1f}, {self.sigma**2:.2f})"""

        def __sampstr__(self):
            return f"""Cluster {self.index} (S) ~ N({self.sample_mu:.1f}, {self.sample_sigma**2:.2f})"""
        def __sampstr2__(self):
            return f"""N({self.sample_mu:.1f}, {self.sample_sigma**2:.2f})"""
        # </editor-fold>

        # <editor-fold desc="PLOTTING METHODS">
        def plot_dists(self, fig = None, ax = None, gen = True, samp = True):
            # Plots both the generative and sample distributions of the Cluster
            plot = False
            if fig is None:
                fig, ax = plt.subplots()
                plot = True
            ax.set_title("[Environment] Cluster Generative and Sample Distributions")
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

    class ArmBelief(object):

        def __init__(self, Cluster, arm_index, mu, sigma):
            self.Cluster = Cluster
            self.index = arm_index
            self.mu = mu
            self.sigma = sigma

        def sample(self, k =1):
            return norm.rvs(loc = self.mu, scale = self.sigma, size = k, random_state=self.Cluster.Environment.rng)

        def __str__(self):
            return f"""ArmBelief{self.index} ~ N({self.mu:.1f}, {self.sigma**2:.2f})"""

