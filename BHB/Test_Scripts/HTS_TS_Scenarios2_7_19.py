import numpy as np

from BHB.Estimators.KnownVarianceEstimator import KnownVarianceEstimator
from BHB.Estimators.UCBEstimator import UCBEstimator
from BHB.Estimators.BayesUCBEstimator import BayesUCBEstimator

from BHB.Policies.HierarchicalThompsonSampling import Full_HTS, Tiered_HTS
from BHB.Policies.UCB import UCB


seed = 123456
rng = np.random.default_rng(seed)
nbClusters = 5
nbArms = 100
Horizon = 5000
nbTrials = 30
nbTrials2 = 30

env_settings1 = {
    'nbClusters' :nbClusters, # Number of Clusters
    "nbArms": nbArms, # Number of arms
    "Horizon": Horizon, # Time horizon
    "hyperprior_means": np.ones(nbClusters)*0,
    "hyperprior_variances": np.ones(nbClusters)*16,
    "cluster_variances": np.ones([nbClusters])*4, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}

env_settings2 = {
    'nbClusters' :nbClusters, # Number of Clusters
    "nbArms": nbArms, # Number of arms
    "Horizon": Horizon, # Time horizon
    "hyperprior_means": np.ones(nbClusters)*0,
    "hyperprior_variances": np.ones(nbClusters)*16,
    "cluster_variances": np.ones([nbClusters])*16, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}

env_settings3 = {
    'nbClusters' :nbClusters, # Number of Clusters
    "nbArms": nbArms, # Number of arms
    "Horizon": Horizon, # Time horizon
    "hyperprior_means": np.ones(nbClusters)*0,
    "hyperprior_variances": np.ones(nbClusters)*16,
    "cluster_variances": np.ones([nbClusters])*25, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}


env_settings4 = {
    'nbClusters' :2, # Number of Clusters
    "nbArms": nbArms, # Number of arms
    "Horizon": Horizon, # Time horizon
    "hyperprior_means": np.ones(2)*0,
    "hyperprior_variances": np.ones(2)*16,
    "cluster_variances": np.ones([2])*4, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}

env_settings5 = {
    'nbClusters' :4, # Number of Clusters
    "nbArms": nbArms, # Number of arms
    "Horizon": Horizon, # Time horizon
    "hyperprior_means": np.ones(4)*0,
    "hyperprior_variances": np.ones(4)*16,
    "cluster_variances": np.ones([4])*4, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}

env_settings6 = {
    'nbClusters' :6, # Number of Clusters
    "nbArms": nbArms, # Number of arms
    "Horizon": Horizon, # Time horizon
    "hyperprior_means": np.ones(6)*0,
    "hyperprior_variances": np.ones(6)*16,
    "cluster_variances": np.ones([6])*4, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}

env_settings7 = {
    'nbClusters' :8, # Number of Clusters
    "nbArms": nbArms, # Number of arms
    "Horizon": Horizon, # Time horizon
    "hyperprior_means": np.ones(8)*0,
    "hyperprior_variances": np.ones(8)*16,
    "cluster_variances": np.ones([8])*4, # Variance for each Cluster Distribution
    "reward_variance": 1, # Reward variance for all rewards
}


policies = {
    "HTS": {"est_priors":None},
    "TS": {"est_priors": None},
}

scenario1 = {
    "nbTrials": nbTrials,
    "policies": policies,
    "env_settings": env_settings1,
    "rng": rng
}

scenario2 = {
    "nbTrials": nbTrials,
    "policies": policies,
    "env_settings": env_settings2,
    "rng": rng
}

scenario3 = {
    "nbTrials": nbTrials,
    "policies": policies,
    "env_settings": env_settings3,
    "rng": rng
}

scenario4 = {
    "nbTrials": nbTrials2,
    "policies": policies,
    "env_settings": env_settings4,
    "rng": rng
}

scenario5 = {
    "nbTrials": nbTrials2,
    "policies": policies,
    "env_settings": env_settings5,
    "rng": rng
}

scenario6 = {
    "nbTrials": nbTrials2,
    "policies": policies,
    "env_settings": env_settings6,
    "rng": rng
}

scenario7 = {
    "nbTrials": nbTrials2,
    "policies": policies,
    "env_settings": env_settings7,
    "rng": rng
}