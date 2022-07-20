import numpy as np



Env1a_Settings = {
    "nbClusters":3, # Number of Clusters
    "nbArms": 100, # Number of arms
    "Horizon": 5_000, # Time horizon
    "hyperprior_means": np.ones(3)*5,
    "hyperprior_variances": np.ones(3)*2,
    "cluster_variances": np.array([1,1,1]), # Variance for each Cluster Generative Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(111), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)
    "Cluster_Dec_Crit": 'best_arm', # Cluster Decision Criteria ('order_stat','threshold', 'best_arm')
    "Cluster_Threshold": None, # Threshold if Cluster_Dec_Crit = 'threshold'
    "HTS_initial_samples": 1, # Number of initial samples of each Cluster before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}

