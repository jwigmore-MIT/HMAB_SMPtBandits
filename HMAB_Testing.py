##
from SMPyBandits.Environment import HMABenv
from SMPyBandits.Arms import UnboundedGaussian
from SMPyBandits.Policies.Posterior import NormalGamma
from SMPyBandits.Policies.HMABPolicy1 import HMABPolicy1
from SMPyBandits.Policies.HMABPolicy2 import HMABPolicy2
from SMPyBandits.Policies.HMABPolicy4 import HMABPolicy4
from SMPyBandits.Policies.HMABPolicy3 import HMABPolicy3
from SMPyBandits.Policies.UCB import  UCB
from SMPyBandits.Environment import Evaluator
import numpy as np

##
nbBands = 2
nbBins = 70
eta0 = 10
gam0 = 1
alpha0 = 1
beta0 = 1

posteriorParams = [(eta0,gam0,alpha0, beta0)]
# posteriorParams = {
#     "mu0": [],
#     "lam0": [],
#     "alpha0": [],
#     "beta0": []
# }

# Policy = HMABPolicy1(nbBands, nbBins, NormalGamma, posteriorParams) # Policy is the agent - doesn't know reward distributions and must maintain estimates

#Band, Bin = Policy.choice()

##
HORIZON = 1000
REPETITIONS = 1
N_JOBS = 1

bandDistribution = "Gaussian"
bandParams = ((9, 1), (11, 1)) # list of (mean_i, variance_i) for the X_i


Band_configuration = {
    "nbBands": nbBands,
    "nbBins": nbBins,
    "bandDistribution": bandDistribution,
    "bandParams": bandParams,
    "binDistribution": UnboundedGaussian
}

ENVIRONMENTS = [{
                 "isHMAB": True,
                 "Band_Config": Band_configuration,
                 "compareMAB": True
                 }] #Need this to fully specify the HMAB problem - used in __initEnvironments to set up arms


# POLICIES = [
#     {
#         "isHMAB": True,
#         "archtype": HMABPolicy1,
#         'params': [nbBands, nbBins, NormalGamma, posteriorParams],
#     },
# {
#         "isHMAB": True,
#         "archtype": HMABPolicy2,
#         'params': [nbBands, nbBins, NormalGamma, posteriorParams],
#     },
# {
#         "isHMAB": True,
#         "archtype": HMABPolicy3,
#         'params': [nbBands, nbBins, NormalGamma, posteriorParams],
#     },
#     {
#         "isHMAB": False,
#         "archtype": UCB,
#         "params": {}
#     }
# ]

# Goal is to populate this for the evaluator function

POLICIES = [
{
        "isHMAB": True,
        "archtype": HMABPolicy2,
        'params': [nbBands, nbBins, NormalGamma, posteriorParams],
    },
{
        "isHMAB": True,
        "archtype": HMABPolicy4,
        'params': [nbBands, nbBins, NormalGamma, posteriorParams],
    },
{
        "isHMAB": False,
        "archtype": UCB,
        "params": {}
    }

]
configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,
    "compareMAB": True
}
eval = Evaluator(configuration)
#
eval.startOneEnv(0,eval.envs[0])
eval.startOneEnv(1, eval.envs[1])


## Plotting Regrets for all in Eval

#eval.plotRegrets([0,1])
#eval.plot_likelihoods(0,0)
# eval.plot_likelihoods(0,1)
# eval.plot_likelihoods(0,2)
##
for i in range(3):
    for j in range(2):
        eval.getBestArm(j,i)

##
eval.plotArmPulls(0,1)
##
# NG = NormalGamma(mu=1, lam = 10, alpha = 1, beta = 2)
# X, T = NG.sample_k(k = 1000)
# EX = np.mean(X)
# ET = np.mean(T)
#
# NG.update(X[0])
#
# #
#
# Policy = HMABPolicy(2, posterior= NormalGamma) # Policy is the agent - doesn't know reward distributions and must maintain estimates
# Policy.getReward(0, 1) #So when we get a reward for an arm, we must tell our agent to update known information
#


##
# N = 2
# M = 5
# Band_Dist = "Gaussian"
# Band_params = ((0.5, 0.05), (0.4, 1)) # list of (mean_i, variance_i) for the X_i
#
#
# configuration = {
#     "N": N,
#     "M": M,
#     "Distribution": Band_Dist,
#     "Band_params": Band_params,
#     "bin_distribution": Gaussian
# }
#
# hmab = HMAB(configuration)