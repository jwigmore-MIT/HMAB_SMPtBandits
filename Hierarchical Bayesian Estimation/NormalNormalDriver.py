from NormalNormalModel import Environment, Estimator
import numpy as np
import matplotlib.pyplot as plt


BAND_MEAN = 5
BAND_SIGMA = 2
NUM_BINS = 10000

BIN_SIGMA = 1



# Bayesian Priors
prior_mu_p = 4
prior_sigma_p = 2
prior_sigma_b = 1



env = Environment(BAND_MEAN, BAND_SIGMA,NUM_BINS, BIN_SIGMA)
estimator = Estimator(env, NUM_BINS, prior_mu_p, prior_sigma_p, prior_sigma_b, BIN_SIGMA)



## Setup
env.plot_band_distribution()
fig, ax = plt.subplots()
ax.cred_int = 0.95
estimator.posterior.plot_posterior(q = 0.005,fig =fig, ax=ax)
estimator.simulate(0,10)
estimator.simulate(1,10)



## Updating check
# First create some sample from the environment



# post_fig, post_ax = plt.subplots(2,1)
# post_ax.cred_p = 0.95
# plot_settings = {
#     "fig": post_fig,
#     "ax": post_ax,
#     "q" : 0.005
# }
# post_ax.set_title("Belief on $\mu_b$")


#
# bin_index = 0
# num_samples = 2
# env_samples = env.sample_bin(bin_index, num_samples)
#
# estimator.update(bin_index, env_samples, verbose = True, plt_set = plot_settings)
#
# # bin_index +=1
# # env_samples = env.sample_bin(bin_index, num_samples)
# # estimator.update(bin_index, env_samples, verbose = True, plt_set = plot_settings)
# #
# #
# #
# # bin_index +=1
# # env_samples = env.sample_bin(bin_index, num_samples)
# # estimator.update(bin_index, env_samples, verbose = True, plt_set = plot_settings)
#
#
# post_fig.legend()