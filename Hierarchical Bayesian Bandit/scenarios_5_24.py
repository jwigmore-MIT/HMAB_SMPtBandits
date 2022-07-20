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
    "Band_Dec_Crit": 'order_stat', # Band Decision Criteria ('order_stat','threshold', 'best_bin')
    "Band_Threshold": None, # Threshold if Band_Dec_Crit = 'threshold'
    "HTS_initial_samples": 1, # Number of initial samples of each Band before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}

Env1b_Settings = {
    "nbBands":3, # Number of Bands
    "nbBins": 100, # Number of bins
    "Horizon": 5_000, # Time horizon
    "band_means": np.array([3.8,4,4.2]), # Mean for each Band Generative Distribution
    "band_variances": np.array([1,1,1]), # Variance for each Band Generative Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(111), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)
    "Band_Dec_Crit": 'order_stat', # Band Decision Criteria ('order_stat' or 'threshold')
    "Band_Threshold": None, # Threshold if Band_Dec_Crit = 'threshold'
    "HTS_initial_samples": 1, # Number of initial samples of each Band before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}

Env2a_Settings = {
    "nbBands":6, # Number of Bands
    "nbBins": 100, # Number of bins
    "Horizon": 5_000, # Time horizon
    "band_means": np.array([2,3,4,5,6,7]), # Mean for each Band Generative Distribution
    "band_variances": np.array([1,1,1,1,1,1]), # Variance for each Band Generative Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(1111), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)
    "Band_Dec_Crit": 'order_stat', # Band Decision Criteria ('order_stat' or 'threshold')
    "Band_Threshold": None, # Threshold if Band_Dec_Crit = 'threshold'
    "HTS_initial_samples": 1, # Number of initial samples of each Band before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}

Env2b_Settings = {
    "nbBands":6, # Number of Bands
    "nbBins": 500, # Number of bins
    "Horizon": 10_000, # Time horizon
    "band_means": np.array([2,3,4,5,6,7]), # Mean for each Band Generative Distribution
    "band_variances": np.array([1,1,1,1,1,1]), # Variance for each Band Generative Distribution
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(1111), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)
    "Band_Dec_Crit": 'order_stat', # Band Decision Criteria ('order_stat' or 'threshold')
    "Band_Threshold": None, # Threshold if Band_Dec_Crit = 'threshold'
    "HTS_initial_samples": 1, # Number of initial samples of each Band before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}
Env0_Settings = {
    "nbBands":1,
    "nbBins": 20,
    "band_means": np.array([5]),
    "band_variances": np.array([1]),
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(111), # Random number generator instantiation
    "UCB_Type": 'classic', # delta, anytime, classic
    "UCB_delta": None, # Confidence interval for UCB(delta)
    "Band_Dec_Crit": 'threshold', # Band Decision Criteria ('order_stat' or 'threshold')
    "Band_Threshold": 7, # Threshold if Band_Dec_Crit = 'threshold'
    "Horizon": 30_000, # Time horizon
    "HTS_initial_samples": 1, # Number of initial samples of each Band before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}



