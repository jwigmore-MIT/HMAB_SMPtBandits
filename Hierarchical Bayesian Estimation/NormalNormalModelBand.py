import numpy as np
from scipy.stats import norm, t, rv_continuous
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

'''
There is going to be a single Band and many bins thus we need to i

'''
class Environment(object):

    def __init__(self, band_means, band_sigmas, num_bins, bin_sigma):
        self.bands = []
        self.num_bands = len(band_means)
        for i in range(self.num_bands):
            self.bands.append(self.Band(band_means[i], band_sigmas[i], num_bins[i], bin_sigma[i]))
        self.band_rvs = [norm(b.sample_params["band_mean"],b.sample_params["band_sigma"]) for b in self.bands]
        self.mixture_model = self.MixtureModel(self.band_rvs)

    def get_q_percent(self, p_star):
        # Returns the percent of each distribution greater than q_star (cdf(q_star)), and q_star
        # Where q_star is computed from the mixture distribution
        q_star = self.mixture_model.ppf(p_star)
        p_i = []
        for band in self.band_rvs:
            p_i.append(band.cdf(q_star))
        return p_i, q_star


    def plot_pdf_and_mixture(self, q_star = None):
        fig, ax = plt.subplots()
        fig, ax = self.plot_bands_distributions(fig = fig, ax  = ax, normalize = True)
        fig, ax = self.mixture_model.plot_pdf(fig, ax, q_star)
        return fig, ax

    def plot_bands_distributions(self, fig = None, ax = None, band_ind = None, normalize = False):
        if fig is None:
            fig, ax = plt.subplots()
        if band_ind is None:
            band_ind = list(range(self.num_bands))
        if normalize:
            weights = 1/self.num_bands
            for ind in band_ind:
                fig, ax, x = self.bands[ind].plot_band_distribution(fig, ax, weights)
        else:
            for ind in band_ind:
                fig, ax, x = self.bands[ind].plot_band_distribution(fig, ax)

        return fig, ax

    class MixtureModel(rv_continuous):
        def __init__(self, submodels, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submodels = submodels

        def _pdf(self, x):
            pdf = self.submodels[0].pdf(x)
            for submodel in self.submodels[1:]:
                pdf += submodel.pdf(x)
            pdf /= len(self.submodels)
            return pdf

        def rvs(self, size):
            submodel_choices = np.random.randint(len(self.submodels), size=size)
            submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
            rvs = np.choose(submodel_choices, submodel_samples)
            return rvs
        def plot_pdf(self, fig = None, ax= None, p_star = None):
            if fig is None:
                fig, ax = plt.subplots()
            alpha = 0.005
            x =  np.linspace(self.ppf(alpha), self.ppf((1-alpha)), 1000)
            pdf = self._pdf(x)
            ax.plot(x, pdf, label = "Mixture")
            ax.set_title("Mixture Distribution over (Sample) Band Distribution")
            if p_star is not None:
                ax.axvline(x = self.ppf(p_star), color = 'k', linestyle = '--', label = f"$(p^*,q^*)$ = ({p_star},{self.ppf(p_star):.2f})")
            ax.legend()
            return fig, ax

        def continuous_bisect_fun_left(self, f, v, lo, hi):
            val_range = [lo, hi]
            k = 0.5 * sum(val_range)
            for i in range(32):
                val_range[int(f(k) > v)] = k
                next_k = 0.5 * sum(val_range)
                if next_k == k:
                    break
                k = next_k
            return k

        # Return the function that is the cdf of the mixture distribution
        def get_mixture_cdf(self):
            ps = 1/len(self.submodels)
            component_distributions = self.submodels
            return lambda x: sum(component_dist.cdf(x) * ps for component_dist in component_distributions)

        # Return the pth quantile of the mixture distribution given by the component distributions and their probabilities

        def _ppf(self, p):
            component_distributions = self.submodels
            mixture_cdf = self.get_mixture_cdf()

            # We can probably be a bit smarter about how we pick the limits
            lo = np.min([dist.ppf(p) for dist in component_distributions])
            hi = np.max([dist.ppf(p) for dist in component_distributions])

            return self.continuous_bisect_fun_left(mixture_cdf, p, lo, hi)





    class Band(object):

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

        def hist_bin_means(self, fig=None, ax=None, weight = 1):
            if fig is None:
                fig, ax = plt.subplots()
            ax.hist(self.bin_means,  density = True, bins='auto', color = "b", alpha = 0.1 )
            #ax.set_title("Bin Means")
            ax.set_xlabel("Bin $\mu$'s")
            #ax.set_ylabel("")
            return fig, ax

        def plot_band_distribution(self, fig = None, ax = None, weights = 1):
            if fig is None:
                fig, ax = plt.subplots()
            ax.set_title("Sample Distribution(s) of Expected Bin Rewards")
            rv = norm(loc = self.sample_params["band_mean"], scale = self.sample_params["band_sigma"])
            x = np.linspace(rv.ppf(0.001), rv.ppf(.999))
            pdf = rv.pdf(x)*weights
            label = f"""N({self.sample_params["band_mean"]:.2f},{self.sample_params["band_sigma"]**2:.2f})"""
            ax.plot(x, pdf, label = label)
            print(f"""Sample Distribution of Expected Bin Rewards : N({self.sample_params["band_mean"]:.2f},{self.sample_params["band_sigma"]**2:.2f})""")
            if weights == 1:
                fig, ax = self.hist_bin_means(fig, ax)
            ax.legend()
            return fig, ax, x


        def plot_band_distribution2(self, bin_index = None, fig = None, ax = None):
            ax.set_title("Sample Distribution of Bin Means")
            rv = norm(loc = self.sample_params["band_mean"], scale = self.sample_params["band_sigma"])
            x = np.linspace(rv.ppf(0.001), rv.ppf(.999))
            pdf = rv.pdf(x)
            ax.plot(x, pdf)
            print(f"""Sample Distribution of Expected Bin Rewards : N({self.sample_params["band_mean"]:.2f},{self.sample_params["band_sigma"]**2:.2f})""")

            fig, ax = self.hist_bin_means(fig, ax)
            if bin_index is not None:
                ax.stem(self.bin_means[bin_index], 0.05, label = f"$\mu_j$ of bin {bin_index}")
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

        def plot_samples(self, bin_index, samples, fig = None, ax = None):
            if fig is None:
                fig, ax = plt.subplots()
            rv = norm(loc=self.bin_means[bin_index], scale=self.bin_sigmas[bin_index])
            x = np.linspace(rv.ppf(0.001), rv.ppf(.999))
            pdf = rv.pdf(x)
            ax.plot(x, pdf, label=f"Bin #{bin_index} Reward Dist ~ N({self.bin_means[bin_index]:.2f}, {self.bin_sigmas[bin_index] ** 2:.2f})")
            ax.stem(samples, np.ones(len(samples))*.1, basefmt = "", label =f"Samples from Bin {bin_index} Reward Dist")
            ax.legend()
            ax.set_title(f"Bin #{bin_index} Reward Dist ~ N({self.bin_means[bin_index]:.2f}, {self.bin_sigmas[bin_index] ** 2:.2f}) + {len(samples)} samples")

        def one_round(self, bin_index, num_samples, plot = False, fig = None, ax = None):
            samples = self.sample_bin(bin_index, num_samples)
            if plot:
                self.plot_samples(bin_index, samples, fig = fig, ax = ax)
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


    def update(self, bin_index, observations, plot = True, fig = None, axs = None, verbose = False):
        # First we add the observations to the correct BinPosterior object

        k_j = len(observations) # number of observations of bin #bin_index in this update
        # Update Reward History
        start = int(self.num_bin_samples[bin_index]) # Start index needed for reward history
        stop = int(start + k_j) # Stop index needed for reward history
        self.reward_history[bin_index, start:stop] = observations # Add observations to reward history
        self.num_bin_samples[bin_index] += k_j # Keeping track of the number of times bin #bin_index has been sampled
        self.sampled_bins.append(bin_index) # Keeping track of what bins have any observations
        self.compute_mle_ests(bin_index) # Computes the MLE Estimate of both mu_b and mu_j
        bin_posterior = self.bin_list[bin_index] # Get the bin posterior object to update
        obs_mean = np.mean(observations) # take the mean of currrent observation vector
        obs_var = np.var(observations) # get the variance of the current observation vector

        # if plot, plot the bayesian priors on mu_b and mu_j
        self.posterior.plot_posterior(q = 0.005, fig = fig, ax = axs["prior_mu_b"])
        bin_posterior.plot_posterior(q = 0.005, fig = fig, ax = axs["prior_mu_j"])

        if verbose:
            print(f"True bin distribution: {self.environment.get_bin_params(bin_index)}")
            print(f"Observation: {observations}")
            print(f"Observation mean: {obs_mean:.3f}")
            print(f"Observation variance: {obs_var:.3f}")
            print(f"Posterior before update: {self.posterior.__str__()}")
            print(f"Bin Posterior before update: {bin_posterior.__str__()}")
        # bin_posterior.add_rewards(observations) Not needed anymore because bin_posterior.update_params() does this
        self.posterior.update(bin_posterior,observations)
        bin_posterior.update_params(self.posterior.mu_b_mmse[-1], observations)
        if verbose:
            print(f"Posterior after the update: {self.posterior.__str__()}")
            print(f"Bin Posterior after update: {bin_posterior.__str__()}")
        # if plot, plot the bayesian posteriors on mu_b, mu_j
        self.posterior.plot_posterior(q = 0.005, fig = fig, ax = axs["post_mu_b"], updated=True)
        bin_posterior.plot_posterior(q = 0.005, fig = fig, ax = axs["post_mu_j"])
        self.plot_mu_b_mle(q = 0.005, fig = fig, ax = axs["freq_mu_b"])
        self.plot_mu_j_mle( bin_index, q=0.01, fig=fig, ax=axs["freq_mu_j"])
        true_mu_b = self.environment.sample_params["band_mean"]
        true_mu_j = self.environment.bin_means[bin_index]
        ii = 1
        for name,ax in axs.items():
            if ii%2 == 0:
                ax.stem(true_mu_j, .1, linefmt="g", markerfmt = "g" ,
                                       label=f"$E[R_j]$ = {true_mu_j:.2f} for j = {bin_index} ")
            else:
                ax.stem(true_mu_b, .1, linefmt="g", markerfmt = "g" ,
                                       label=f"$E[B]$= {true_mu_b:.2f}")
            ax.legend()
            ii+=1



    def compute_mle_ests(self, bin_index):
        self.bin_mle_est[bin_index] = np.mean(self.reward_history[bin_index, :int(self.num_bin_samples[bin_index])])
        self.bin_mle_var[bin_index] = np.var(self.reward_history[bin_index,:int(self.num_bin_samples[bin_index])])
        self.band_mle_est.append(np.mean(self.bin_mle_est[self.sampled_bins]))
        self.band_mle_var.append(np.var(self.bin_mle_est[self.sampled_bins]))
        # Next we update the Posterior Distribution

    def plot_mu_b_mle(self,q = 0.01, fig = None, ax = None):
        if fig is None:
            fig, ax = plt.subplots()
        mu = self.band_mle_est[-1]
        sigma = self.environment.input_params["band_sigma"]
        df = len(self.sampled_bins) - 1
        if df > 1:
            rv = t(df = len(self.sampled_bins), loc = mu , scale = sigma)
            x = np.linspace(rv.ppf(q), rv.ppf(1 - q))
            ci = t.interval(alpha=0.95, df = len(self.sampled_bins),  loc = mu, scale = self.environment.input_params["band_sigma"])
            if max(ci) < mu + 5:
                ax.stem([ci[0], ci[1],mu], np.ones(3)*.4, label = f"0.95 Confidence Interval at ({ci[0]:.2f}, {ci[1]:.2f})")
                pdf = t.pdf(x, df=len(self.sampled_bins), loc=mu, scale=sigma)
                ax.plot(x, pdf, alpha=0.4, label = f"$\hat\mu_b$ t-distribution)")
            else:
                ax.stem(mu, .4, label=f"0.95 Confidence Interval at ({ci[0]:.2f}, {ci[1]:.2f})")
                ax.set_title(
                    f"MLE Estimate of $\mu_b$ from observations with confidence intervals ({ci[0]:.2f}, {ci[1]:.2f})")
        else:
            ax.stem(mu, .4, label=f"MLE Estimate of $\hat\mu_b$ = {mu:.2f}")
            ax.set_title(
                f"MLE Estimate of $\hat\mu_b$={mu:.2f} from single $\hat\mu_j$ observation")
        ax.legend()

    def plot_mu_j_mle(self, bin_index, q = 0.01, fig = None, ax = None):
        if fig is None:
            fig, ax = plt.subplots()
        mu = self.bin_mle_est[bin_index]
        sigma = self.environment.bin_sigmas[bin_index]
        df = int(self.num_bin_samples[bin_index] - 1)
        if df > 1:
            rv = t(df = df, loc = mu , scale = sigma)
            x = np.linspace(rv.ppf(q), rv.ppf(1 - q))
            ci = t.interval(alpha=0.95, df = df,  loc = mu, scale = sigma)
            if max(ci) < mu + 5:
                ax.stem([ci[0], ci[1],mu], np.ones(3)*.4, label = f" ({ci[0]:.2f}, {mu:.2f}, {ci[1]:.2f})")
                pdf = t.pdf(x, df=df, loc=mu, scale=sigma)
                ax.plot(x, pdf, alpha=0.4, label = f"$\hat\mu_j$ t-distribution)")
                ax.set_title(
                    f"MLE Estimate of $\mu_j$={mu:.2f} from observations with confidence intervals ({ci[0]:.2f}, {ci[1]:.2f})")
            else:
                ax.stem(mu, .4, label=f"0.95 Confidence Interval at ({ci[0]:.2f}, {ci[1]:.2f})")
                ax.set_title(
                    f"MLE Estimate of $\mu_j$={mu:.2f} from observations with confidence intervals ({ci[0]:.2f}, {ci[1]:.2f})")
        else:
            ax.stem(mu, .4, label=f"MLE Estimate of $\hat\mu_j$ = {mu:.2f}")
            ax.set_title(
                f"MLE Estimate of $\hat\mu_j$={mu:.2f} from single reward observation")
        ax.legend()



    def simulate(self, bin_index, num_samples):
        fig, axs = plt.subplot_mosaic(
            [['true_band_dist', 'true_rew_dist'],['prior_mu_b', 'prior_mu_j'], ['post_mu_b', "post_mu_j"], ["freq_mu_b", "freq_mu_j"],
             ], sharex=True)
        self.environment.plot_band_distribution2(bin_index = bin_index, fig = fig, ax = axs['true_band_dist'])
        print(f"Drawing {num_samples} samples from bin # {bin_index}")
        samples = self.environment.one_round(bin_index,num_samples, plot = True, fig = fig, ax = axs['true_rew_dist'])
        print(f"Observations: {samples}")

        print("Computing posteriors based on observations")

        self.update(bin_index,samples, verbose=True, fig= fig, axs = axs)
        return fig, axs




    class Posterior:

        def __init__(self, prior_mu_p, prior_sigma_p):
            # Initialize prior parameters that will be "private"
            self.prior = True
            # Initilize Posterior parameters as none
            self.mu_p = [prior_mu_p]
            self.sigma_p = [prior_sigma_p]
            self.num_updates = 0
            self.mu_b_mmse = [self.mu_p[0]]



        def update(self, bin_posterior, observations): # update from observations from a single bin
            self.prior = False
            k_j = len(observations)
            # We don't want to update self.sigma_p yet because we need that value to update self.mu_p
            new_sigma_p = (1/(self.sigma_p[-1]**2) + k_j/(k_j*bin_posterior.sigma_b[-1]**2 + bin_posterior.sigma_j**2))**(-1/2)
            new_mu_p = (new_sigma_p**2)*(self.mu_p[-1]/(self.sigma_p[-1]**2) + (np.sum(observations)/(k_j*bin_posterior.sigma_b[-1]**2 + bin_posterior.sigma_j**2)))
            self.mu_p.append(new_mu_p)
            self.sigma_p.append(new_sigma_p)
            self.mu_b_mmse.append(self.mu_p[-1])
            self.num_updates +=1



        def __str__(self):
            return f"N({self.mu_p[-1]:.3f}, {self.sigma_p[-1]**2:.3f})"

        def plot_posterior(self, q = 0.005, fig = None, ax = None, updated = False):
            if fig is None:
                fig, ax = plt.subplots()
            rv = norm(loc = self.mu_p[-1], scale = self.sigma_p[-1])
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
                if updated:
                    ax.set_title("Bayesian Belief on $\mu_b$ after most recent observations (i.e. Posterior Distribution)")
                else:
                    ax.set_title(("Bayesian Belief on $\mu_b$ before most recent observations"))
            ax.plot(x, pdf, label = label)
            ax.stem([cred_int[0], self.mu_p[-1], cred_int[1]], [.1, .1,.1], label =f"({cred_int[0]:.2f} , {self.mu_p[-1]:.2f}, {cred_int[1]:.2f})")
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
            self.sigma_b = [sigma_b]
            self.sigma_j = sigma_j  # known parameter, doesn't change
            self.rewards = []  # stores most recent rewards for this bin
            self.reward_history = [] #stores all rewards observed for this bin
            ### MMSE estimates all start out the same before any observations
            self.mu_b_mmse = [self.posterior.mu_b_mmse[0]] # these are almost inputs
            self.mu_j_mmse = [self.mu_b_mmse]
            self.num_updates = 0

        def update_params(self, mu_b,new_rewards):
            self.rewards = new_rewards
            self.reward_history.extend(new_rewards)
            k_j = len(new_rewards)
            self.mu_b_mmse.append(mu_b)  # input used to compute current distribution mean

            new_sigma_b = ((1 / (self.sigma_b[-1] ** 2)) + k_j / (self.sigma_j ** 2))**(-1/2)
            #new_sigma_b = ((self.sigma_b**2 + self.sigma_j**2)/(k_j*self.sigma_b**2+self.sigma_j**2))**(-1/2)
            new_mu_b = new_sigma_b ** 2 * (
                        mu_b/ (self.sigma_b[-1] ** 2) + np.sum(self.rewards) / (self.sigma_j ** 2))

            self.sigma_b.append(new_sigma_b)
            self.mu_j_mmse.append(new_mu_b)
            self.num_updates+=1



        def __str__(self):
            return f"P(mu_j|mu_b_mmse = {self.mu_b_mmse[-1]}) = N({self.mu_j_mmse[-1]}, {self.sigma_b[-1] ** 2})"

        def plot_posterior(self, q=0.005, fig=None, ax=None):
            if fig is None:
                fig, ax = plt.subplots()
            rv = norm(loc=self.mu_b_mmse[-1], scale=self.sigma_b[-1])
            x = np.linspace(rv.ppf(q), rv.ppf(1 - q))
            pdf = rv.pdf(x)
            cred_p = 0.95
            cred_q = (1 - cred_p) / 2
            cred_int = [rv.ppf(cred_q), rv.ppf(1 - cred_q)]  # .95 credible interval
            if self.num_updates < 1:
                label = "Prior Distribution on $\mu_j$"
                ax.set_title("Prior Distribution on $\mu_j | \mu_b^{(MMSE)}$")
            else:
                label = f"Update #{self.num_updates}"
                ax.set_title("Bayesian Belief on $\mu_j$ given $\mathbf{r}_j$ and updated $\mu_b^{(MMSE)}$")
            ax.plot(x, pdf, label=label)
            ax.stem([cred_int[0], self.mu_b_mmse[-1], cred_int[1]], [.1, .1,.1], label =f"({cred_int[0]:.2f} , {self.mu_b_mmse[-1]:.2f}, {cred_int[1]:.2f})")

            ax.legend()

        def plot_reward_est(self, q=0.005, fig=None, ax=None):
            if fig is None:
                fig, ax = plt.subplots()
            rv = norm(loc=self.mu_j_mmse[-1], scale=self.sigma_j)
            x = np.linspace(rv.ppf(q), rv.ppf(1 - q))
            pdf = rv.pdf(x)
            ax.set_title("Estimate of reward distribution based on Bayesian $\mu_j^{(MMSE)}$")
            ax.plot(x, pdf)
