from BHB.PG_Environment import *
from BHB.Estimators.KnownVarianceEstimator import KnownVarianceEstimator
from BHB.Estimators.UCBEstimator import UCBEstimator
from BHB.Estimators.BayesUCBEstimator import BayesUCBEstimator
from BHB.Evaluator import *
from BHB.Policies.HierarchicalThompsonSampling import Full_HTS, Tiered_HTS
from BHB.Policies.UCB import  UCB
from UCB_Testing_scenarios import Basic_5
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
#from analysistools import *
import addcopyfighandler


PICKLE_PATH = "/BHB/Generated_Environments"
time = datetime.now().strftime("%Y-%m-%d_%H_%M")
print(time)
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

Settings = Basic_5
print("Running Test with: ")
print(Settings)

# Estimator Parameters

# Evaluator Settings
horizon = Settings["Horizon"]

Env = PG_Environment(Settings) # initialize environment
Env.plot_clusters_dists(gen= False, samp = True)
est_priors = {
    "hyperprior_means": np.zeros(5),
    "hyperprior_variances": np.ones(5)*1e9,
}

Est1 = BayesUCBEstimator(Env, Settings)
Est2 = UCBEstimator(Env, Settings, type_overide="delta")
Est3 = UCBEstimator(Env, Settings, type_overide="MOSS")
Est4 = KnownVarianceEstimator(Env)


Pol1 = UCB(Est1, name = 'UCB - Bayes')
Pol2 = UCB(Est2, name = 'UCB - Delta')
Pol3 = UCB(Est3, name = 'UCB - MOSS')
Pol4 = Full_HTS(Est4, name = 'Full_HTS_Known_Prior')


Policies = [Pol1, Pol2, Pol3, Pol4]

Eval = Evaluator(Env, Policies)


Eval.play_games()


## Regret Plotting
Eval.plot_cumulative_regrets()


## Printing Results
Eval.print_summaries()


## Decision History Plotting
Eval.plot_decision_histories()











# #results_UCB.plot_sorted_decision_history()
# #results_HTS.plot_sorted_decision_history()
# #post_rvs = results_HTS.plot_round_param(int(Settings["Horizon"]-1))
# # post_rvs = results_HTS.plot_round_param(1000)
# #chosen_freqs = results_HTS.simulate_error_prob(post_rvs)
# post_rvs = results_HTS.plot_round_param(int(Settings["Horizon"]-1))
# post_rvs2 = results_HTS.plot_round_param(int(results_HTS.dom_time))
#
# results_HTS.plot_stoch_dom()
# chosen_freqs = results_HTS.simulate_error_prob(post_rvs)
# chosen_freqs2 = results_HTS.simulate_error_prob(post_rvs2)
#
# post_rvs3 = results_HTS.plot_round_param(int(20))
# chosen_freqs3 = results_HTS.simulate_error_prob(post_rvs3)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
