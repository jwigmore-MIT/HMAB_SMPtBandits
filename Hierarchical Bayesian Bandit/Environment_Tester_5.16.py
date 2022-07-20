import numpy as np
from Environment import *
from Policy import *
from Estimator import *
from Evaluator import *
from scenarios_old import *
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from analysistools import *

from config import DEBUGSETTINGS as DS
import pickle
import addcopyfighandler

PICKLE_PATH = "C:\GitHub\HMAB\Hierarchical Bayesian Bandit\Environments"
time = datetime.now().strftime("%Y-%m-%d_%H_%M")
print(time)
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

Settings = Env1c_Settings

# Estimator Parameters
q_star = Settings["Band_Threshold"]



delta = Settings["UCB_Confidence"]# UCB confidence level

# Evaluator Settings
horizon = Settings["Horizon"]

Env = Environment(Settings) # initialize environment
Env.plot_bands_dists(gen= False, samp = True, threshold = q_star)
Env.compute_tails(q_star)



Est1 = KnownVarianceEstimator(Env)
Ucb_Est = UCBEstimator(Env, delta)


Pol1 = HierarchicalThompsonSampling(Est1, q_star, initial_samples= Settings["HTS_initial_samples"], name="Thompson/Point", exp_frac = Settings["HTS_exp_frac"])
Pol2 = UCB1(Ucb_Est, name="UCB1")
#
Policies = [Pol1, Pol2]

Eval = Evaluator(Env, Policies)



results_HTS = Eval.play_game(horizon, env_id=0, pol_id=0)
results_HTS.summarize_results(bins = True)
results_HTS.store_results(f"C:\GitHub\HMAB\Hierarchical Bayesian Bandit\Results\\5.17\\KnownVariance_{time}.pdf")
#plt.close('all')
results_HTS.plot_decision_history()



results_UCB = Eval.play_game(horizon, env_id=0, pol_id=1, results_type="UCB")
results_UCB.plot_decision_history()

fig, ax = plt.subplots()

results_HTS.plot_cumulative_regret(ax = ax)
results_UCB.plot_cumulative_regret(ax = ax)

post_rvs = results_HTS.plot_round_param(int(Settings["Horizon"]-1))
# post_rvs = results_HTS.plot_round_param(1000)
chosen_freqs = results_HTS.simulate_error_prob(post_rvs)




















