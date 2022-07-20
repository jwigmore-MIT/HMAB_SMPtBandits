import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

'''
There is going to be a single Band and many bins thus we need to i

'''


class Environment(object):

    def __init__(self, band_mean, band_sigma, num_bins, bin_sigma):
        """
        "Ground Truth" of the simulation i.e. what we are trying to estimate
        :param band_mean: mean of the band
        :param band_sigma: standard deviation of the band
        :param num_bins: number of bins within the band
        :param bin_sigma: standard deviation of each bin within the band
        """
        self.input_params = {
            "band_mean": band_mean,
            "band_sigma": band_sigma
        }
        self.num_bins = num_bins
        self.bin_means = norm.rvs(loc=band_mean, scale=band_sigma,
                                  size=num_bins)  # Generate bin means from the Band Generative distribution
        self.bin_sigmas = [bin_sigma] * num_bins
        # Sample distribution of bin means may differ from input due to finite sample size (i.e. num_bins < \infty )
        self.sample_params = {
            "band_mean": np.mean(self.bin_means),
            "band_sigma": np.std(self.bin_means)
        }

    def hist_bin_means(self, fig=None, ax=None):
        if fig is None:
            fig, ax = plt.subplots()
        ax.hist(self.bin_means,  density = True, bins='auto', color = "b", alpha = 0.3 )
        #ax.set_title("Bin Means")
        ax.set_xlabel("Bin $\mu$'s")
        #ax.set_ylabel("")
        return fig, ax

    def plot_band_distribution(self, fig = None, ax = None):
        if fig is None:
            fig, ax = plt.subplots()
        ax.set_title("Sample Distribution of Bin Means")
        rv = norm(loc = self.sample_params["band_mean"], scale = self.sample_params["band_sigma"])
        x = np.linspace(rv.ppf(0.001), rv.ppf(.999))
        pdf = rv.pdf(x)
        ax.plot(x, pdf)
        print(f"""Sample Distribution of Expected Bin Rewards : N({self.sample_params["band_mean"]:.2f},{self.sample_params["band_sigma"]**2:.2f})""")

        fig, ax = self.hist_bin_means(fig, ax)
        return fig, ax, x

    def plot_bin_band_dist(self, bin_index = None, fig = None, ax = None):
        if fig is None:
            fig, ax = plt.subplots(2,1, sharex=True)
        if bin_index is None:
            bin_index = np.random.randint(0,self.num_bins)
        fig, ax[0], x = self.plot_band_distribution(fig, ax[0])
        ax[0].stem(self.bin_means[bin_index], 0.1)
        print(f"Plotting distribution of bin # {bin_index}")
        rv = norm(loc = self.bin_means[bin_index], scale = self.bin_sigmas[bin_index])
        #x = np.linspace(rv.ppf(0.001), rv.ppf(.999))
        pdf = rv.pdf(x)
        ax[1].plot(x, pdf, label = f"Bin #{bin_index} Reward Dist ~ N({self.bin_means[bin_index]:.2f}, {self.bin_sigmas[bin_index]**2:.2f})")
        ax[1].legend()
        return fig, ax


    def sample_bin(self, bin_index, samples):
        return norm.rvs(loc = self.bin_means[bin_index], scale = self.bin_sigmas[bin_index], size = samples)

    def get_bin_params(self, bin_index):
        return f"Bin #{bin_index} ~ N({self.bin_means[bin_index]:.3f}, {self.bin_sigmas[bin_index]})"

    def plot_samples(self, bin_index, samples):
        fig, ax = plt.subplots()
        rv = norm(loc=self.bin_means[bin_index], scale=self.bin_sigmas[bin_index])
        x = np.linspace(rv.ppf(0.001), rv.ppf(.999))
        pdf = rv.pdf(x)
        ax.plot(x, pdf, label=f"Bin #{bin_index} Reward Dist ~ N({self.bin_means[bin_index]:.2f}, {self.bin_sigmas[bin_index] ** 2:.2f})")
        ax.stem(samples, np.ones(len(samples))*.1, basefmt = "", label =f"Samples from Bin {bin_index} Reward Dist")
        ax.legend()
        ax.set_title(f"Bin #{bin_index} Reward Dist ~ N({self.bin_means[bin_index]:.2f}, {self.bin_sigmas[bin_index] ** 2:.2f}) + {len(samples)} samples")

    def one_round(self, bin_index, num_samples, plot = False):
        samples = self.sample_bin(bin_index, num_samples)
        if plot:
            self.plot_samples(bin_index, samples)
        return samples


class Estimator(object):

    def __init__(self, environment, num_bins, prior_mu_p, prior_sigma_p, prior_sigma_b, sigma_j):
        self.environment = environment
        self.bin_list = []
        self._prior_mu_p = prior_mu_p
        self._prior_sigma = prior_sigma_p
        self._sigma_b = prior_sigma_b  # prior standard deviation of the bin posterior
        self._sigma_j = sigma_j  # known standard deviation of the reward likelihood
        # Initialize the Posterior object using the prior parameters
        self.posterior = self.Posterior(prior_mu_p, prior_sigma_p)

        # Since all bins should be initialized the same except for their index
        # create an instance of the bin object and initialize it
        # Then just copy everything from it, set the index, and then add to bin_list
        temp_bin = self.BinPosterior(self.posterior, prior_sigma_b, sigma_j)
        self.reward_history = np.zeros([num_bins, 100]) #100 is the max number of samples per bin
        self.num_bin_samples = np.zeros(num_bins)
        self.bin_mle_est = np.zeros(num_bins)
        self.bin_mle_var = np.zeros(num_bins)
        self.band_mle_est = []
        self.band_mle_var = []
        self.sampled_bins = []
        for n in range(num_bins):
            bin_n = deepcopy(temp_bin)
            bin_n.index = n
            self.bin_list.append(bin_n)
        print("Estimator object created")


    def update(self, bin_index, observations, plot = True, verbose = False):
        # First we add the observations to the correct BinPosterior object
        if plot:
            fig, axs = plt.subplot_mosaic([['prior_mu_b', 'prior_mu_j'], ['post_mu_b', "post_mu_j"], ["freq_mu_b", "freq_mu_j"]])
        k_j = len(observations)
        start = int(self.num_bin_samples[bin_index])
        stop = int(start + k_j)
        self.reward_history[bin_index, start:stop] = observations
        self.num_bin_samples[bin_index] += k_j
        self.sampled_bins.append(bin_index)
        self.compute_mle_ests(bin_index)
        bin_posterior = self.bin_list[bin_index]
        obs_mean = np.mean(observations)
        obs_var = np.var(observations)
        if plt_set is None:
            pass
        elif self.posterior.num_updates == 0:
            self.posterior.plot_posterior(plt_set["q"], plt_set["fig"], plt_set["ax"])
        if verbose:
            print(f"True bin distribution: {self.environment.get_bin_params(bin_index)}")
            print(f"Observation: {observations}")
            print(f"Observation mean: {obs_mean:.3f}")
            print(f"Observation variance: {obs_var:.3f}")
            print(f"Posterior before update: {self.posterior.__str__()}")
            print(f"Bin Posterior before update: {bin_posterior.__str__()}")
        # bin_posterior.add_rewards(observations) Not needed anymore because bin_posterior.update_params() does this
        self.posterior.update(bin_posterior,observations)
        bin_posterior.update_params(self.posterior.mu_b_mle, observations)
        if plt_set is None:
            pass
        else:
            self.posterior.plot_posterior(plt_set["q"], plt_set["fig"], plt_set["ax"])
        if verbose:
            print(f"Posterior after the update: {self.posterior.__str__()}")
            print(f"Bin Posterior after update: {bin_posterior.__str__()}")

    def compute_mle_ests(self, bin_index):
        self.bin_mle_est[bin_index] = np.mean(self.reward_history[bin_index, :int(self.num_bin_samples[bin_index])])
        self.bin_mle_var[bin_index] = np.var(self.reward_history[bin_index,:int(self.num_bin_samples[bin_index])])
        self.band_mle_est.append(np.mean(self.bin_mle_est[self.sampled_bins]))
        self.band_mle_var.append(np.var(self.bin_mle_est[self.sampled_bins]))
        # Next we update the Posterior Distribution

    def simulate(self, bin_index, num_samples):
        print(f"Drawing {num_samples} samples from bin # {bin_index}")
        samples = self.environment.one_round(bin_index,num_samples, plot = True)
        print(f"Observations: {samples}")

        print("Computing posteriors based on observations")

        self.update(bin_index,samples, verbose=True)




    class Posterior:

        def __init__(self, prior_mu_p, prior_sigma_p):
            # Initialize prior parameters that will be "private"
            self.prior = True
            # Initilize Posterior parameters as none
            self._mu_p = prior_mu_p
            self._sigma_p = prior_sigma_p
            self.mu_b_mle = prior_mu_p  # initial MLE estimate of mu_b prior to any observations

            self.sigma_p = self._sigma_p # Current sigma_p = None before any observations
            self.mu_p = self._mu_p #
            self.num_updates = 0


        def update(self, bin_posterior, observations): # update from observations from a single bin
            self.prior = False
            k_j = len(observations)
            # We don't want to update self.sigma_p yet because we need that value to update self.mu_p
            new_sigma_p = (1/(self.sigma_p**2) + k_j/(k_j*bin_posterior.sigma_b**2 + bin_posterior.sigma_j**2))**(-1/2)
            new_mu_p = (new_sigma_p**2)*(self.mu_p/(self.sigma_p**2) + (np.sum(observations)/(k_j*bin_posterior.sigma_b**2 + bin_posterior.sigma_j**2)))
            self.mu_p = new_mu_p
            self.sigma_p = new_sigma_p
            self.mu_b_mle = self.mu_p
            self.num_updates +=1



        def __str__(self):
            return f"N({self.mu_p:.3f}, {self.sigma_p**2:.3f})"

        def plot_posterior(self, q = 0.005, fig = None, ax = None):
            if fig is None:
                fig, ax = plt.subplots()
            rv = norm(loc = self.mu_p, scale = self.sigma_p)
            x = np.linspace(rv.ppf(q), rv.ppf(1-q))
            pdf = rv.pdf(x)
            if hasattr(ax,"cred_p"):
                cred_p = ax.cred_p
            else:
                cred_p = 0.95
            cred_q = (1-cred_p)/2
            cred_int = [rv.ppf(cred_q),rv.ppf(1-cred_q)] # .95 credible interval
            if self.num_updates <1:
                label = "Prior Distribution on $\mu_b$"
                ax.set_title("Prior Distribution on $\mu_b$")
            else:
                label = f"Update #{self.num_updates}"
                ax.set_title("Bayesian Belief on $\mu_b$ given reward history")
            ax.plot(x, pdf, label = label)
            ax.stem(cred_int, [.1,.1], label =f"{cred_p} Credible Interval")
            ax.legend()




    class BinPosterior:
        """
        This is the distribution P(mu_j|mu_b) meaning its only well defined given some input mu_b
        """

        def __init__(self, posterior, sigma_b, sigma_j):
            """

            :param posterior: posterior object that contains the parameters of the posterior (and prior) distribution
            :param B: known mean of the bin's posterior distribution
            """
            self.index = None  # indicates bin number
            self.posterior = posterior
            # prior parameter, will not change but will have a posterior (sigma_b) that depends # on it
            self._sigma_b = sigma_b
            self.sigma_j = sigma_j  # known parameter, doesn't change
            self.rewards = []  # stores most recent rewards for this bin
            self.reward_history = [] #stores all rewards observed for this bin
            self.sigma_b = self._sigma_b  # Note this parameter can be updated based on
            self.mu_b = self.posterior.mu_b_mle
            self.mu_b_prime = None
            #self.update_params(self.posterior.mu_b_mle, [])

        def update_params(self, mu_b_prime,new_rewards):
            self.rewards = new_rewards
            self.reward_history.extend(new_rewards)
            k_j = len(new_rewards)
            self.mu_b_prime = mu_b_prime  # input used to compute current distribution mean

            new_sigma_b = ((1 / (self._sigma_b ** 2)) + k_j / (self.sigma_j ** 2))**(-1/2)
            #new_sigma_b = ((self.sigma_b**2 + self.sigma_j**2)/(k_j*self.sigma_b**2+self.sigma_j**2))**(-1/2)
            new_mu_b = new_sigma_b ** 2 * (
                        mu_b_prime / (self._sigma_b ** 2) + np.sum(self.rewards) / (self.sigma_j ** 2))

            self.sigma_b = new_sigma_b
            self.mu_b = new_mu_b

        # def add_rewards(self, new_rewards):
        #     self.rewards = new_rewards
        #     self.reward_history.extend(new_rewards)

        def __str__(self):
            return f"P(mu_j|mu_b_mle = {self.mu_b_prime}) = N({self.mu_b}, {self.sigma_b ** 2})"
