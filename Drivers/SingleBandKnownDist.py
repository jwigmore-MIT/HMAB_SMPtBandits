import numpy as np
import matplotlib.pyplot as plt

from SMPyBandits.Distribution import *
from SMPyBandits.Environment import *
from SMPyBandits.Policies.HMABStopping.SingleBandSearch import *
from SMPyBandits.Environment import Evaluator
from SMPyBandits.Environment.HMAB_Eval import *

"""
Testing scenario where there is a single band with N (very large, approximately infinite) bins
We know the Band Distribution
But we don't know the Bin Distribution
"""

nbBins = 10000
percentile = .80 # p i.e. set percentage of environment that we want the chosen bin to be better than
HORIZON = 10000 # Upperbound on the Horizon
confidence = 0.95 # confidence level we want in our objective

binConfig = {
    "binDistribution": UnboundedGaussian,
    "binParams": 1 # Standard deviation for unbounded gaussians
}

bandConfig = {
    "nbBins": nbBins,
    "bandDistribution": Normal, # Distribution family of each Z_i
    "bandParams": {
        "mu": [5], # each element is Mu_i for Z_i ~ N(Mu_i, Sigma_i^2)
        "std": [1], # Sigma_i for above
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
} #Configures the environment

envOpt = {
    "Gen": True,
    "BF": False,
    "Bin": False,
    "Index" : 0, # to select the color
} ## For plots from the environment

DistributionSetup = {
    "BandDist": Normal,
    "InitBandParams": (bandConfig["bandParams"]["mu"], bandConfig["bandParams"]["std"]), # Setting our initial estimate of Band parameters to the true parameters
    "BinDist": Normal,
    "InitBinParams": (0,1)
} ## For setting up distribution estimates in Policy

AlgoParams = {
    "percentile": percentile,
    "confidence": {"bins":confidence},

}

POLICIES = [
    {
        "isHMAB": True,
        "archtype": SingleBandSearch,
        "params": [nbBands, nbBins, DistributionSetup, True, AlgoParams]
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
eval.plotChoices(0,0)
fpol = eval.final_policy[0][0]
fig, ax = plt.subplots()
ax = fpol.Bands[0].plotSampledBins(ax)
fig.show()

printChosenBinInfo(eval, 0,0)
printAllBinInfo(eval,0,0)

#Pol = SingleBandSearch(nbBands, nbBins, DistributionSetup, verbose  = True, AlgoParams = AlgoParams)





# Gaussian Setup #
