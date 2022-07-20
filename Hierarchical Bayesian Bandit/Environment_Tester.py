import numpy as np
from Environment import *
from Policy import *
from Estimator import *
from Evaluator import *
import addcopyfighandler

from config import DEBUGSETTINGS as DS






nbBands = 3
nbBins = 500

band_variance = 20
band_sigma = np.sqrt(band_variance)

band_means = [3,5,7]
band_variance = np.ones(nbBands)*20
band_params = []
sigmas_i = []
for b in range(nbBands):
    band_params.append([band_means[b],np.sqrt(band_variance[b])])
    sigmas_i.append(np.sqrt(band_variance[b]))

# Environment Parameters
bin_params = 1

# Estimator Parameters
sigma_r = 1 # Known Bin standard deviation

# Threshold
percentile = 0.95
q_star = 7

# Evaluator Settings
horizon = 1000

Env = Environment(nbBands, nbBins, band_params, bin_params) # initialize environment
Env.plot_bands_dists()

Est1 = KnownVarianceEstimator(Env, sigmas_i, sigma_r, percentile = percentile) # initialize estimator with no prior estimates\


Pol1 = HierarchicalThompsonSampling(Est1, nbBands, nbBins, q_star, name = "Thompson")
#Pol1 = ChooseOne(Est, nbBands, nbBins, q_star, band = 0, bin = 0, samples = 1, name = 'ChooseOne')
Pol2 = ChooseRandom(Est1, nbBands, nbBins, q_star, bin = 0, samples = 1, name = "ChooseRandom" )
Policies = [Pol1, Pol2]

Eval = Evaluator(Env, Policies)

results1 = Eval.play_game(horizon, env_id = 0, pol_id = 0)
# results_UCB.plot_posterior_history(band_indices=[0,1,2])
# results_UCB.plot_posterior_variance_history(band_indices=[0,1,2])
results1.plot_decision_history()
results1.print_final_posterior()
results1.plot_band_param_history()
results1.summarize_results()

# (results2, environment2, policy2, estimator2) = Eval.play_game(horizon, 0,1,0)
# results2.plot_posterior_history(band_indices=[0,1,2])
# results2.plot_posterior_variance_history(band_indices=[0,1,2])



