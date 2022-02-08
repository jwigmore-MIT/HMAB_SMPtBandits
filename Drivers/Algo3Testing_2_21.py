import numpy as np
import matplotlib.pyplot as plt

from SMPyBandits.Distribution import *
from SMPyBandits.Environment import *
from SMPyBandits.Policies import HMABPolicy
from SMPyBandits.Policies import HMABAlgo3
from SMPyBandits.Environment import Evaluator







nbBins = 1000
nbBands = 3
percentile = .99
P1 = 0.2
m = 10
HORIZON = 1000
eps = 0.1
confidence = 0.95
# Gaussian Setup #

binConfig = {
    "binDistribution": UnboundedGaussian,
    "binParams": 1 # Standard deviation for unbounded gaussians
}

bandConfig = {
    "nbBins": nbBins,
    "bandDistribution": Normal, # Distribution family of each Z_i
    "bandParams": {
        "mu": [3, 5, 7], # each element is Mu_i for Z_i ~ N(Mu_i, Sigma_i^2)
        "std": [1, 1, 1], # Sigma_i for above
    },
    "binConfig": binConfig,
    "percentile": percentile
}




nbBands = bandConfig["bandParams"]["mu"].__len__()

envConfig = {
    "isHMAB": True,
    "nbBands": nbBands,
    "nbBins": nbBins,
    "bandConfig": bandConfig,
    "binConfig": binConfig,
    "compareMAB": True,
    "verbose": True
}

envOpt = {
    "Gen": True,
    "BF": True,
    "Bin": True,
    "Index" : 0, # to select the color
}
# env = HMABenv(envConfig)
#
# fig, ax = plt.subplots()
# ax.envOpt = envOpt
# env.plotDists(ax)
# ax.legend(loc = 'best', frameon = False)
# fig.show()

BayesSetup = {
    "hyperprior": NormalGamma,
    "hyperparameters": (1,1,1,1),
    "BandDist": Normal,
    "InitBandParams": (0,1),
    "BinDist": Normal,
    "InitBinParams": (0,1)
}


AlgoParams = {
    "T": HORIZON,
    "m": m,
    "P1": P1,
    "percentile": percentile,
    "eps": eps,
    "confidence": confidence
}

# Pol = HMABAlgo1(nbBands, nbBins, BayesSetup, AlgoParams = AlgoParams)

POLICIES = [
    {
        "isHMAB": True,
        "archtype": HMABAlgo3,
        "params": [nbBands, nbBins, BayesSetup, True, AlgoParams]
    }
]

ENVIRONMENTS = [
    {
        "isHMAB": True,
        "envConfig": envConfig,
        "compareMAB": False
    }
]

configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": 1,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": 1,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,
    "compareMAB": False
}

eval = Evaluator(configuration)
eval.startAllEnv()

eval.printSummary(0,0,3)
eval.plotChoices(0,0)


# fig, ax = plt.subplots()
# ax.envOpt = envOpt
# eval.envs[0].plotDists(ax)
# ax.legend(loc = 'best', frameon = False)
# fig.show()

###
pol = eval.final_policy[0][0]
result = eval.final_result[0][0]
arms_sorted_pulls = np.argsort(-pol.pulls)
most_pulled_bin = pol.Bins[arms_sorted_pulls[0]]
chosen_bin = pol.Bins[pol.bestArms[0]]