# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from SMPyBandits.Environment import MAB, Evaluator, tqdm
from SMPyBandits.Arms import Bernoulli
from SMPyBandits.Policies import *
import matplotlib as mpl


HORIZON = 1000
REPETITIONS = 10
N_JOBS = 1

ENVIRONMENTS = [  # 1)  Bernoulli arms
        {   # A very easy problem, but it is used in a lot of articles
            "arm_type": Bernoulli,
            "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        {   # An other problem, best arm = last, with three groups: very bad arms (0.01, 0.02), middle arms (0.3 - 0.6) and very good arms (0.78, 0.8, 0.82)
            "arm_type": Bernoulli,
            "params": [0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.795, 0.8, 0.805]
        },
        {   # A very hard problem, as used in [Cappé et al, 2012]
            "arm_type": Bernoulli,
            "params": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.1]
        },
    ]

POLICIES = [
        # --- UCB1 algorithm
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 1
            }
        },
        {
            "archtype": UCBalpha,
            "params": {
                "alpha": 0.5  # Smallest theoretically acceptable value
            }
        },
        # --- Thompson algorithm
        {
            "archtype": Thompson,
            "params": {}
        },
        # --- KL algorithms, here only klUCB
        {
            "archtype": klUCB,
            "params": {}
        },
        # --- BayesUCB algorithm
        {
            "archtype": BayesUCB,
            "params": {}
        },
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
}
evaluation = Evaluator(configuration)

for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
    # Evaluate just that env
    evaluation.startOneEnv(envId, env)



def plotAll(evaluation, envId):
    evaluation.printFinalRanking(envId)
    evaluation.plotRegrets(envId)
    evaluation.plotRegrets(envId, semilogx=True)
    evaluation.plotRegrets(envId, meanReward=True)
    evaluation.plotBestArmPulls(envId)

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.4, 7)
_ = plotAll(evaluation, 0)
