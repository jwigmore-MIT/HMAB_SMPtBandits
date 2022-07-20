import numpy as np
from Environment import *
from Policy import *
from Estimator import *
from Evaluator import *
from scenarios_old import *
from matplotlib.backends.backend_pdf import PdfPages

from config import DEBUGSETTINGS as DS
import pickle
import addcopyfighandler

PICKLE_PATH = "C:\GitHub\HMAB\Hierarchical Bayesian Bandit\Environments"

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()



# Estimator Parameters
percentile = 0.95
q_star = 7

# Evaluator Settings
horizon = 500

Env = Environment(Env5_Settings) # initialize environment
Env.plot_bands_dists(samp = False, threshold = q_star)

Env.plot_bands_dists(gen = False, threshold= q_star)

Est1 = KnownVarianceEstimator(Env, percentile = percentile)

#Est2 = ModalBayesEstimator(Env, percentile = percentile)


Pol1 = HierarchicalThompsonSampling(Est1, q_star, initial_samples= 2, name="Thompson/Point")
#Pol2 = HierarchicalThompsonSampling(Est2, q_star, initial_samples= 2, name="Thompson/Modal")
#
Policies = [Pol1]

Eval = Evaluator(Env, Policies)

#results_UCB = Eval.play_game(horizon, env_id = 0, pol_id = 0, plot = True)
#results_UCB.summarize_results()

results0 = Eval.play_game(horizon, env_id=0, pol_id=0)
results0.summarize_results(bins = True)
results0.store_results("C:\GitHub\HMAB\Hierarchical Bayesian Bandit\Results\\KnownVariance.pdf")


# results2 = Eval.play_game(horizon, env_id=0, pol_id=2)
# results2.summarize_results()
# results2.store_results("C:\GitHub\HMAB\Hierarchical Bayesian Bandit\Results\\ChooseRandomPoint2.pdf")
# plt.close('all')



# results_UCB = Eval.play_game(horizon, env_id=0, pol_id=1)
# results_UCB.summarize_results()
# results_UCB.store_results("C:\GitHub\HMAB\Hierarchical Bayesian Bandit\Results\\HierThompsonModal2.pdf")
# plt.close('all')
# results2.plot_decision_history()
# results2.plot_bands_param_history()

# results3 = Eval.play_game(horizon, env_id=0, pol_id=3)
# results3.summarize_results()
# results3.store_results("C:\GitHub\HMAB\Hierarchical Bayesian Bandit\Results\\ChooseRandomPointModal2.pdf")
# plt.close('all')






