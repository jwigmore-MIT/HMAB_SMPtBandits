import numpy as np



Env1a_Settings = {
    "nbClusters":5, # Number of Clusters
    "nbArms": 30, # Number of arms
    "Horizon": 5_00, # Time horizon
    "hyperprior_means": np.ones(5)*5,
    "hyperprior_variances": np.ones(5)*2,
    "cluster_variances": np.ones([5]), # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(10001), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)

}


Env1b_Settings = {
    "nbClusters":5, # Number of Clusters
    "nbArms": 50, # Number of arms
    "Horizon": 5_000, # Time horizon
    "hyperprior_means": np.ones(5)*5,
    "hyperprior_variances": np.ones(5)*5,
    "cluster_variances": np.ones([5])*2, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(5031997), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)

}

Env1c_Settings = {
    "nbClusters":5, # Number of Clusters
    "nbArms": 50, # Number of arms
    "Horizon": 5_000, # Time horizon
    "hyperprior_means": np.ones(5)*20,
    "hyperprior_variances": np.ones(5)*25,
    "cluster_variances": np.ones([5])*4, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(5031997), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)

}

Env2a_Settings = { # Different prior for estimator
    "nbClusters":5, # Number of Clusters
    "nbArms": 20, # Number of arms
    "Horizon": 5_00, # Time horizon
    "hyperprior_means": np.ones(5)*20,
    "hyperprior_variances": np.ones(5)*25,
    "cluster_variances": np.ones([5])*4, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(5031997), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)

}
