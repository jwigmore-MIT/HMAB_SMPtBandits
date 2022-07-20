import numpy as np



Env1_Settings = {
    "nbBands":3,
    "nbBins": 500,
    "band_means": np.array([1,2,3]),
    "band_variances": np.array([1,1,1]),
    "reward_variance": 1,
    "rng": np.random.default_rng(0),
    "UCB_Confidence": 0.05,
    "Band_Dec_Crit": 'threshold',
    "Band_Threshold": 4,
    "Horizon": 5_00,
    "HTS_initial_samples": 1,
    "HTS_exp_frac": 0,
}

Env1b_Settings = {
    "nbBands":3,
    "nbBins": 100,
    "band_means": np.array([1,2,3]),
    "band_variances": np.array([1,1,1])*2,
    "reward_variance": 1,
    "rng": np.random.default_rng(0),
    "UCB_Confidence": 0.05,
    "Band_Threshold": 4,
    "Horizon": 5_000,
    "HTS_initial_samples": 1,
    "HTS_exp_frac": 0,
}

Env1c_Settings = {
    "nbBands":3,
    "nbBins": 100,
    "band_means": np.array([1,2,3]),
    "band_variances": np.array([1,1,1]),
    "reward_variance": 1,
    "rng": np.random.default_rng(0),
    "UCB_Confidence": 0.05,
    "Band_Threshold": 4,
    "Horizon": 5_000,
    "HTS_initial_samples": 1,
    "HTS_exp_frac": 0,
}

Env2_Settings = {
    "nbBands":3,
    "nbBins": 100,
    "band_means": np.array([1,1,1]),
    "band_variances": np.array([1,1,1]),
    "reward_variance": 1,
    "rng": np.random.default_rng(0),
    "UCB_Confidence": 0.05,
    "Band_Dec_Crit": 'order_stat',
    "Band_Threshold": None,
    "Horizon": 5_000,
    "HTS_initial_samples": 1,
    "HTS_exp_frac": 0,
}

Env2b_Settings = {
    "nbBands":3,
    "nbBins": 500,
    "band_means": np.array([1,1,1]),
    "band_variances": np.array([1,1,1]),
    "reward_variance": 1,
    "rng": np.random.default_rng(0),
    "UCB_Confidence": 0.05,
    "Band_Dec_Crit": 'order_stat',
    "Band_Threshold": None,
    "Horizon": 5_000,
    "HTS_initial_samples": 1,
    "HTS_exp_frac": 0,
}



Env1_Settings = {
    "nbBands":3,
    "nbBins": 500,
    "band_means": np.array([1,2,3]),
    "band_variances": np.array([1,1,1]),
    "reward_variance": 1, # Reward variance for all rewards
    "rng": np.random.default_rng(0), # Random number generator instantiation
    "UCB_Confidence": 0.05, # Confidence interval for UCB(delta)
    "Band_Dec_Crit": 'threshold', # Band Decision Criteria ('order_stat' or 'threshold')
    "Band_Threshold": 4, # Threshold if Band_Dec_Crit = 'threshold'
    "Horizon": 5_00, # Time horizon
    "HTS_initial_samples": 1, # Number of initial samples of each Band before running algorithm
    "HTS_exp_frac": 0, # Forced Exploration percentage for Thompson Sampling
    "stoch_dom": 0.01 # Stochastic Dominance Percentage - if not None Policy will compute stochastic Dominance
}




#
# Env1_Settings = {
#     "nbBands":1,
#     "nbBins": 500,
#     "band_means": np.array([5]),
#     "band_variances": np.array([2]),
#     "reward_variance": 1,
#     "rng": np.random.default_rng(0)
#
# }
#
#
# Env2_Settings = {
#     "nbBands" :3,
#     "nbBins": 100,
#     "band_means": np.array([4,5,6]),
#     "band_variances": np.array([2,2,2]),
#     "reward_variance": 1,
#     "rng": np.random.default_rng(1)
# }
#
#
# Env3_Settings = {
#     "nbBands" :3,
#     "nbBins": 500,
#     "band_means": np.array([5,5,5]),
#     "band_variances": np.array([1,2,3]),
#     "reward_variance": 1,
#     "rng": np.random.default_rng(3)
# }
#
# Env4_Settings = {
#     "nbBands" :3,
#     "nbBins": 100,
#     "band_means": np.array([3,5,7]),
#     "band_variances": np.array([4,4,4]),
#     "reward_variance": 1,
#     "rng": np.random.default_rng(3)
# }
#
# Env5_Settings = {
#     "nbBands" :3,
#     "nbBins": 100,
#     "band_means": np.array([4,5,6]),
#     "band_variances": np.array([8,4,1]),
#     "reward_variance": 1,
#     "rng": np.random.default_rng(3)
# }
#
# Env6_Settings = {
#     "nbBands" :2,
#     "nbBins": 100,
#     "band_means": np.array([4,5]),
#     "band_variances": np.array([2,2]),
#     "reward_variance": 1,
#     "rng": np.random.default_rng(0)
# }
#
#
# Env7_Settings = {
#     "nbBands" :6,
#     "nbBins": 500,
#     "band_means": np.array([1,2,3, 4, 5, 6]),
#     "band_variances": np.array([1,1,1,1,1,1])*2,
#     "reward_variance": 1,
#     "rng": np.random.default_rng(3)
# }