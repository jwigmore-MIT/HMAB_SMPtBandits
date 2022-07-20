import numpy as np

from BHB.Estimators.KnownVarianceEstimator import KnownVarianceEstimator
from BHB.Estimators.UCBEstimator import UCBEstimator
from BHB.Estimators.BayesUCBEstimator import BayesUCBEstimator

from BHB.Policies.HierarchicalThompsonSampling import Full_HTS, Tiered_HTS
from BHB.Policies.UCB import UCB


seed = 123456
rng = np.random.default_rng(seed)
nbClusters = 5
env_settings = {
    'nbClusters' :nbClusters, # Number of Clusters
    "nbArms": 60, # Number of arms
    "Horizon": 4_000, # Time horizon
    "hyperprior_means": np.ones(nbClusters)*10,
    "hyperprior_variances": np.ones(nbClusters)*4,
    "cluster_variances": np.ones([nbClusters])*2, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}

policies = {
    "Full HTS": {"est_priors":None},
    "Full HTS UI": {"est_priors":None},
    "Full Sample HTS": {"est_priors":None},
    "Tiered HTS": {"est_priors":None},
    "Tiered HTS UI": {"est_priors":None},
    "Bayes UCB": {"est_priors": None},
    "MOSS UCB": {"UCB_Type": "MOSS"},
    "Delta UCB": {"UCB_Type": "delta","UCB_delta": 0.05}


}

multi_scenario = {
    "nbTrials": 5,
    "policies": policies,
    "env_settings": env_settings,
    "rng": rng
}