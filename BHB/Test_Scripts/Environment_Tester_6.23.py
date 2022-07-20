from BHB.PG_Environment import *
from BHB.Estimators.KnownVarianceEstimator import KnownVarianceEstimator
from BHB.Estimators.UCBEstimator import UCBEstimator
from BHB.Evaluator import *
from BHB.Policies.HierarchicalThompsonSampling import Full_HTS, Tiered_HTS
from BHB.Policies.UCB1 import  UCB1
from scenarios_6_23 import *
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

Settings = Env2a_Settings
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

Est1 = KnownVarianceEstimator(Env)
Est2 = KnownVarianceEstimator(Env)
Est3 = KnownVarianceEstimator(Env, est_priors)
Est4 = KnownVarianceEstimator(Env)
Est5 = UCBEstimator(Env, Settings)

Pol1 = Full_HTS(Est1, name = 'Full_HTS_Forced_Sample')
Pol2 = Full_HTS(Est2, name = 'Full_HTS_Known_Prior', sample_all= False)
Pol3 = Full_HTS(Est3, name = 'Full_HTS_Uninformative', sample_all= False)
Pol4 = Tiered_HTS(Est4, name = 'Tiered_HTS')
Pol5 = UCB1(Est5, name = 'UCB1')

Policies = [Pol1, Pol2, Pol3, Pol4, Pol5]

Eval = Evaluator(Env, Policies)


results_FHTS_A = Eval.play_game(horizon, env_id=0, pol_id=0)
results_FHTS_B = Eval.play_game(horizon, env_id=0, pol_id=1)
results_FHTS_C = Eval.play_game(horizon, env_id=0, pol_id=2)
results_THTS = Eval.play_game(horizon, env_id=0, pol_id=3)
results_UCB = Eval.play_game(horizon, env_id=0, pol_id=4, results_type= 'UCB')

## Regret Plotting
fig, ax = plt.subplots()
results_FHTS_A.plot_cumulative_regret(ax = ax)
results_FHTS_B.plot_cumulative_regret(ax = ax)
results_FHTS_C.plot_cumulative_regret(ax = ax)
results_THTS.plot_cumulative_regret(ax = ax)
results_UCB.plot_cumulative_regret(ax = ax)

## Printing Results
results_FHTS_A.summarize_results(arms = True)
results_FHTS_B.summarize_results(arms = True)
results_FHTS_C.summarize_results(arms = True)
results_THTS.summarize_results(arms = True)
results_UCB.summarize_results(arms = True)

## Decision History Plotting
results_FHTS_A.plot_decision_history()
results_FHTS_B.plot_decision_history()
results_FHTS_C.plot_decision_history()
results_THTS.plot_decision_history()
results_UCB.plot_decision_history()










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
