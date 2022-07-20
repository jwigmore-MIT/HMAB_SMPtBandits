import numpy as np


Env1a_Settings = {
    "nbBands":3, # Number of Bands
    "nbBins": 100, # Number of bins
    "Horizon": 5_000, # Time horizon
    "band_means": np.array([3,4,5]), # Mean for each Band Generative Distribution
    "band_variances": np.array([1,1,1]), # Variance for each Band Generative Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(111), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)
    "Band_Dec_Crit": 'best_bin', # Band Decision Criteria ('order_stat','threshold', 'best_bin')
    "Band_Threshold": None, # Threshold if Band_Dec_Crit = 'threshold'
    "HTS_initial_samples": 1, # Number of initial samples of each Band before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}

