##
from SMPyBandits.Environment import HMABenv
from SMPyBandits.Arms import UniformArm
from SMPyBandits.Policies import BandThenBin1
from SMPyBandits.Policies import BandThenBin2
from SMPyBandits.Distribution.Uniform import Uniform

from SMPyBandits.Policies.UCB import  UCB
from SMPyBandits.Environment import Evaluator

## Environment setup
nbBands = 2
nbBins = 50
bandScale = 4.5 # For uniform distribution - the width of the band distributions
binScale = 10 # The width of the bin distributions
# Lets start by creating a Uniform Environment
bandDistribution = "Uniform"
bandParams = {
    "min": [3, 10],
    'scale': [bandScale, bandScale, bandScale, bandScale]
}
binConfig = {
    "scale": binScale
}
bandConfig = {
    "nbBins": nbBins,
    "bandDistribution": bandDistribution,
    "bandParams": bandParams,
    "binDistribution": UniformArm,
    "binConfig": binConfig
}


envConfig = {
    "isHMAB": True,
    "bandConfig": bandConfig,
    "compareMAB": True,
    "nbBands": nbBands,
    "nbBins": nbBins,
    "verbose": False
}
## Environment Creation Test
env = HMABenv(envConfig)


## Policy Setup
priorParams = {
    "loc": 7.5,
    "scale": bandScale
} #<dict> with first guess on location and scale

## Policy Creation Test

policy = BandThenBin1(nbBands, nbBins, Uniform, priorParams, verbose= False)

## Evaluator test
HORIZON = 1000
REPETITIONS = 1
N_JOBS = 1
ENVIRONMENTS = [{
    "isHMAB": True,
    "envConfig": envConfig,
    "compareMAB": True
}]
POLICIES = [
    {
        "isHMAB": True,
        "archtype": BandThenBin2,
        "params": [nbBands, nbBins, Uniform, priorParams, True]
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

## Running policies

eval.startOneEnv(0, eval.envs[0])
eval.startOneEnv(1,eval.envs[1])

eval.printHMABResults(0,0)
##
eval.plotRewards([0,1], [0,1])

##
eval.plot_likelihoods(envId = 0, policyID = 0)

## Plotting total number of arm pulls
eval.plotArmPulls(0,0)
eval.plotArmPulls(1,1)
eval.plotRegrets([0,1])
