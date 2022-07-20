# Local Imports
from .Distribution import Distribution
from scipy.stats import norm
import numpy as np

from numpy.random import gamma, normal


class NormalGamma(Distribution):

    def __init__(self, omega, zeta, alpha, beta):
        r"""Create a posterior assuming the prior is : N(omega, tau_i^(-1))..
        """
        
        # Normal Parameters
        self._omega = float(omega)  # initial value
        self.omega = float(omega)  #: Parameter :math:`\omega` of the posterior
        self._zeta = float(zeta)  # initial value
        self.zeta = float(zeta)  #: The parameter :math:`\sigma` of the posterior

        # Gamma Parameters
        self._alpha = float(alpha)
        self.alpha = float(alpha)
        self._beta = float(beta)
        self.beta = float(beta)

        # internal memories
        self._nb_obs = 0  # number of observations!
        self._sum_data = 0.0  # sum of observations!


    def __str__(self, init = False):
        if init:
            return "NormalGamma(_omega = {:.3g},_zeta = {:.3g}, _alpha =  {:.3g}, _beta = {:.3g})".format(self._omega, self._zeta,
                                                                                                self._alpha, self._beta)
        else:
            return "NormalGamma(omega = {:.3g},zeta = {:.3g}, alpha =  {:.3g}, beta = {:.3g})".format(self.omega,
                                                                                                      self.zeta,
                                                                                                      self.alpha,
                                                                                                      self.beta)


    def reset(self, *args):
        r""" Reset the for parameters :omega and tau_i, as when creating a new NormalGamma posterior."""
        if not args:
            self.omega = self._omega
            self.zeta = self._zeta
            self.alpha = self._alpha
            self.beta = self._beta
        else:
            self.omega = args[0]
            self.zeta = args[1]
            self.alpha = args[2]
            self.beta = args[3]

    def sample(self):
        # https://en.wikipedia.org/wiki/Normal-zetama_distribution
        # Random variable X|T ~ N(omega, 1/(zeta*T) where T|(alpha, beta) ~ Gamma(alpha, beta)
        # Then the joint distribtuion (X,T) ~ NormalGamma(omega, zeta, alpha, beta)
        # Sample first from gamma(alpha, beta) to get T (tau_i)
        # Sample from normal(omega, 1/(lzeta*T)) to get X
        T = gamma(self.alpha, 1. / self.beta)
        var = (1 / (T * self.zeta))
        X = normal(loc=self.omega, scale=np.sqrt(var))
        return (X, T)

    def sample_k(self, k=100):
        X = np.zeros((k, 1))
        T = np.zeros((k, 1))
        for ii in range(k):
            X[ii], T[ii] = self.sample()
        return X, T

    def getInitParams(self): # Return the initial parameters
        return self._omega, self._zeta, self._alpha, self._beta

    def getVariance(self):  # Variance of the X and T values for the Normal Gamma Distribution
        if self.alpha == 1:  # When alpha
            varX = np.inf
        else:
            varX =  (self.beta / (self.zeta * (self.alpha - 1)))
        varT = self.alpha*self.beta**(-2)

        return varX, varT

    def getExpVal(self): #returns the mean of the distribution
        # When this distribution is a posterior, the mean is the Bayes Estimate under the Mean Squared Error Risk function
        evX = self.omega
        evT = self.alpha/self.beta
        return evX, evT

    def update(self, obs):  # This observation is an np array of observations or a single value
        # Normal Gamma updates are now based on n-samples instead of 1

        if isinstance(obs,np.ndarray):
            self._nb_data += obs.shape[0]
            # self._sum_data += float(obs)
            x = np.mean(obs)
            s = np.var(obs)
            n = obs.shape[0]

            omega_new = (self.zeta * self.omega + n * x) / (self.zeta + n)
            zeta_new = self.zeta + n
            alpha_new = self.alpha + n / 2
            beta_new = self.beta + 1 / 2 * (n * s + self.zeta * n * (x - self.omega) ** 2 / (self.zeta + n))
            self.omega = omega_new
            self.zeta = zeta_new
            self.alpha = alpha_new
            self.beta = beta_new
            self.setLLParams()
        else:
            self._nb_data += 1
            self._sum_data += float(obs)
            omega_new = (self.zeta * self.omega + obs) / (self.zeta + 1)
            zeta_new = self.zeta + 1
            alpha_new = self.alpha + 1 / 2
            beta_new = self.beta + 1 / 2 * (self.zeta * (obs - self.omega) ** 2 / (self.zeta + 1))
            self.omega = omega_new
            self.zeta = zeta_new
            self.alpha = alpha_new
            self.beta = beta_new




    def plotDist(self, ax):
        pass # No need to plot the 3D Normal Gamma Distribution at the moment
        # plt.rcParams.update({"text.usetex": True,})
        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        # markers = makemarkers(self.nbPolicies)
        #
        # if issubclass(policy.posteriorDist, Uniform):
        #     for Band in env.X:
        #         # Get true distribution
        #         c = colors[Band.i]
        #         i = Band.i
        #         X_i = Band.X_i  # Frozen r.v. representing the distribution of E[lam_ij]
        #         ub = X_i.ppf(1) + .01  # Upper Bound for linspace
        #         lb = X_i.ppf(0) - .01  # Lower Bound for linspace
        #
        #         n = np.linspace(lb, ub, 500)
        #         y = X_i.pdf(n)
        #         ax[0].plot(n, y, '-', color=c,
        #                    label=f'Band ({Band.i}) ~ {X_i.dist.name}(loc={X_i.kwds["loc"]}, scale = {X_i.kwds["scale"]})')
        #         arms = Band.lam
        #         y_arms = np.ones(len(arms)) * np.max(y)
        #         # ax[0].hist(arms, bins = env.nbArms*50, color=c, density=False, histtype='bar', label=f"Empirical ArmBelief Distribution", alpha=1)
        #         binWidth = self.cfg['environment'][envId]["envConfig"]["bandConfig"]["binConfig"][
        #             "scale"]  # Need to implement this better lol
        #         ax[0].bar(arms, y_arms, width=binWidth, edgecolor=c, label=f"Empirical ArmBelief Distribution", fill=True,
        #                   alpha=0.2)
        #
        #         ax[0].legend(loc='best', frameon=False)
        #
        #         X_init = uniform(loc=policy.posteriorBand[i]._min, scale=policy.posteriorBand[i]._scale)
        #         ub1 = X_init.ppf(1) + .01  # Upper Bound for linspace
        #         lb1 = X_init.ppf(0) - .01  # Lower Bound for linspace
        #         n1 = np.linspace(lb1, ub1, 500)
        #         c2 = colors[Band.i * 2]
        #         if i == 0:
        #             ax[1].plot(n1, X_init.pdf(n1), '--', color='black',
        #                        label=f'Prior Estimate ~ {X_init.dist.name}(loc={X_init.kwds["loc"]}, scale = {X_init.kwds["scale"]})')
        #
        #         X_est = uniform(loc=policy.likelihoodParams[i]["loc"], scale=policy.likelihoodParams[i]["scale"])
        #         ub2 = X_est.ppf(1) + .01  # Upper Bound for linspace
        #         lb2 = X_est.ppf(0) - .01  # Lower Bound for linspace
        #         n2 = np.linspace(lb2, ub2, 500)
        #         ax[1].plot(n2, X_est.pdf(n2), '-', color=c2,
        #                    label=f'Band ({Band.i}) ~ {X_est.dist.name}(loc={X_est.kwds["loc"]:.3f}, scale = {X_est.kwds["scale"]:.3f})')
        #
        # ax[0].legend(loc='best', frameon=False)
        # ax[1].legend(loc='best', frameon=False)
        # fig.suptitle(f"Bands: {env.nbBands}, Bins: {env.nbBins}")
        # fig.show()
        #
        #
        #
