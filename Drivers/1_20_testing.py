## Imports
from SMPyBandits.Arms import UnboundedGaussian
from SMPyBandits.Distribution import NormalGamma

from SMPyBandits.Policies.HMABOld import HMABSampling1
from SMPyBandits.Policies.UCB import  UCB
from SMPyBandits.Environment import Evaluator

## Environment setup
nbBins = 50
nbBands = 3

# Gaussian Setup #
bandConfig = {
    "nbBins": nbBins,
    "bandDistribution": "Gaussian", # Distribution family of each Z_i
    "bandParams": {
        "mu": [3, 5, 7], # each element is Mu_i for Z_i ~ N(Mu_i, Sigma_i^2)
        "std": [1, 1, 1], # Sigma_i for above
    },
    "binDistribution": UnboundedGaussian, # Bin distribution i.e. distribution of R_ij
    "binConfig": {
        "std": 1, # Controls the standard deviation of R_ij
        # Means come from sampling Z_i for each R_ij
    }
}

envConfig = {
    "isHMAB": True,
    "nbBands": nbBands,
    "nbBins": nbBins,
    "bandConfig": bandConfig,
    "compareMAB": True,
    "verbose": False
}

# env = HMABenv(envConfig)

## Evaluator Setup

priorParams = {
    "eta": 5,
    "gam": 1,
    "alpha": 1,
    "beta": 1,
}
POLICIES = [
    {
        "isHMAB": True,
        "archtype": HMABSampling1,
        "params": [nbBands, nbBins, NormalGamma, priorParams, False, 1]
    },
# {
#         "isHMAB": True,
#         "archtype": HMABThompson1,
#         "params": [nbBands, nbBins, NormalGamma, priorParams, True, 2]
#     },
# {
#         "isHMAB": True,
#         "archtype": HMABThompson1,
#         "params": [nbBands, nbBins, NormalGamma, priorParams, True, 3]
#     },
    {
      "isHMAB": False,
      "archtype": UCB,
      "params": {}
    }
]

ENVIRONMENTS = [{
    "isHMAB": True,
    "envConfig": envConfig,
    "compareMAB": True
}]

configuration = {
    # --- Duration of the experiment
    "horizon": 10000,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": 1,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": 1,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,
    "compareMAB": True
}

eval = Evaluator(configuration)

eval.startAllEnv()

#eval.printHMABResults(0,0)
eval.plotTrueDistribution(envId=0)
eval.plotEmpDistribution(envId=0)
for n in range(POLICIES.__len__()-1):
    eval.plotEstDistributions(envId = 0, policyId = n)
for n in range(POLICIES.__len__()-1):
    eval.evalEstimates(envId= 0, policyId=n)





eval.plotRegrets([0,1])