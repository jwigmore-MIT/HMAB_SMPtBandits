import numpy as np



Basic_5 = {
    "nbClusters":5, # Number of Clusters
    "nbArms": 10, # Number of arms
    "Horizon": 5_000, # Time horizon
    "hyperprior_means": np.ones(5)*5,
    "hyperprior_variances": np.ones(5)*2,
    "cluster_variances": np.ones([5]), # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(99), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": 0.95, # Confidence interval for UCB(delta)

}


