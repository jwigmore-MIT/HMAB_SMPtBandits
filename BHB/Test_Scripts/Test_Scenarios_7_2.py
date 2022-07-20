import numpy as np



Basic_5 = {
    "nbClusters":5, # Number of Clusters
    "nbArms": 30, # Number of arms
    "Horizon": 5_000, # Time horizon
    "hyperprior_means": np.ones(5)*5,
    "hyperprior_variances": np.ones(5)*2,
    "cluster_variances": np.ones([5]), # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(10001), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)

}


multi_setup = {
    "nbClusters":5, # Number of Clusters
    "nbArms": 30, # Number of arms
    "Horizon": 5_000, # Time horizon
    "hyperprior_means": np.ones(5)*5,
    "hyperprior_variances": np.ones(5)*2,
    "cluster_variances": np.ones([5]), # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(10001), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)
}


