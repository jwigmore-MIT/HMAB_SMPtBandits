## Imports
from SMPyBandits.Arms import UnboundedGaussian
from SMPyBandits.Distribution import NormalGamma

from SMPyBandits.Policies import HMABSampling1
from SMPyBandits.Policies import HMABPercentile2
from SMPyBandits.Policies.UCB import  UCB
from SMPyBandits.Environment import Evaluator

## Environment setup
nbBins = 100
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

nbBands = bandConfig["bandParams"]["mu"].__len__()

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
    "eta": 6,
    "gam": 1,
    "alpha": 1,
    "beta": 20,
}

percentile = .99
POLICIES = [
#     {
#         "isHMAB": True,
#         "archtype": HMABThompson1,
#         "params": [nbBands, nbBins, NormalGamma, priorParams, False, 1]
#     },
# {
#         "isHMAB": True,
#         "archtype": HMABThompson1,
#         "params": [nbBands, nbBins, NormalGamma, priorParams, False, 2]
#     },
{
        "isHMAB": True,
        "archtype": HMABSampling1,
        "params": [nbBands, nbBins, NormalGamma, priorParams, False, 1]
    },
{
        "isHMAB": True,
        "archtype": HMABPercentile2,
        "params": [nbBands, nbBins, NormalGamma, priorParams, False, 1, percentile]
    },
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
    "horizon": 50000,
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
## Uncomment
#eval.startAllEnv()

#eval.printHMABResults(0,0)
eval.plotTrueDistribution(envId=0)
eval.plotEmpDistribution(envId=0)

## Uncomment
# for n in range(POLICIES.__len__()-1):
#     eval.plotEstDistributions(envId = 0, policyId = n)
# for n in range(POLICIES.__len__()-1):
#     eval.evalEstimates(envId= 0, policyId=n)
# regAlg1 = eval.getCumulatedRegret(0,0)[-1]
# regAlg2 = eval.getCumulatedRegret(1,0)[-1]
# regUCB1= eval.getCumulatedRegret(2,1)[-1]
# imp1 = 1/(regAlg1/regUCB1)
# imp2 = 1/(regAlg2/regUCB1)
#eval.plotRegrets([0,1])




# eval.plotEstDistributions(envId = 0, policyId = 0)
# eval.plotEstDistributions(envId = 0, policyId = 1)
# # eval.plotEstDistributions(envId = 0, policyId = 2)
# result1 = eval.evalEstimates(envId= 0, policyId=0)
# result2 = eval.evalEstimates(envId= 0, policyId=1)
# # result3 = eval.evalEstimates(envId= 0, policyId=2)




