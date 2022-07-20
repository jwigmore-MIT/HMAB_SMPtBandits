

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

import pandas as pd


class UCB_Results(object):

    def __init__(self, environment, policy, horizon, r_name):
        self.environment = environment
        self.nbBands = self.environment.nbBands
        self.nbBands = self.environment.nbBins
        self.horizon = horizon
        self.policy = policy
        self.name = r_name
        self.decision_history= np.zeros([horizon, 3])
        self.time = 0
        self.pre_estimator = deepcopy(policy.estimator)
        self.post_estimator = None
        self.latest_params = None # NOT IMPLEMENTED

    def store(self, band_index, bin_index, observations, latest_params= None):
        if isinstance(observations, (np.ndarray, tuple, list)):
            num_obs = observations.__len__()
            self.decision_history[self.time:self.time + num_obs, 0] = band_index
            self.decision_history[self.time:self.time + num_obs, 1] = bin_index
            self.decision_history[self.time:self.time + num_obs, 2] = observations
            self.time += num_obs

    def plot_decision_history(self):
        nbBands = self.environment.nbBands
        fig, axes = plt.subplots(nbBands,1)
        y= np.zeros([self.decision_history.shape[0], nbBands]) - 10
        x = range(self.decision_history.shape[0])
        for b in range(nbBands):
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Band"] == 0]
            rounds = (self.decision_history["Band"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds,b] = self.decision_history[rounds]["Bin"].to_numpy()
            if nbBands > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"B{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title(f"[{self.name}] Decision History")
                axes[b].set_ylabel("Channel Index")
                axes[b].set_xlabel("Round t")

            else:
                axes.scatter(x=x, y=y[:, b], color=color, label=f"B{b}")
                axes.set_ylim(0)
                axes.legend()
                axes.set_title(f"[{self.name}] Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def plot_sorted_decision_history(self):
        nbBands = self.environment.nbBands
        fig, axes = plt.subplots(nbBands,1)
        y= np.zeros([self.decision_history.shape[0], nbBands]) - 10
        x = range(self.decision_history.shape[0])
        for b in range(nbBands):
            sort_index = self.environment.bands[b].sort_index
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Band"] == 0]
            rounds = (self.decision_history["Band"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds, b] = sort_index[[int(i) for i in self.decision_history[rounds]["Bin"].values]]
            if nbBands > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"B{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title(f"[{self.name}] Decision History - Sorted")
                axes[b].set_ylabel("Sorted Channel Index")
                axes[b].set_xlabel("Round t")

            else:
                axes.scatter(x=x, y=y[:, b], color=color, label=f"B{b}")
                axes.set_ylim(0)
                axes.legend()
                axes.set_title(f"[{self.name}] Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def finalize_results(self):
        exp_reward = [self.get_expected_reward(int(band), int(bin)) for band, bin in self.decision_history[:,:2]]
        self.decision_history = np.concatenate([self.decision_history,np.array([exp_reward]).T], axis=1)
        self.decision_history = pd.DataFrame(self.decision_history, columns=["Band", "Bin", "Reward", "Exp Reward"])
        self.compute_regret()

    def compute_regret(self):
        opt_reward = np.max(self.environment.best_means)
        real_regret = opt_reward - self.decision_history['Reward']
        expected_regret = opt_reward - self.decision_history["Exp Reward"]

        real_regret_sum = real_regret.cumsum()
        expected_regret_sum = expected_regret.cumsum()

        self.regret = pd.concat({"Real": real_regret, "Expected": expected_regret}, axis = 1, )
        self.sum_regret = pd.concat({"Real": real_regret_sum, "Expected": expected_regret_sum,}, axis = 1)


    def get_expected_reward(self, band, bin):
        return self.environment.bin_means[band][bin]

    def plot_cumulative_regret(self, fig = None, ax = None, emp = False):
        if ax is None:
            fig, ax = plt.subplots()
        if emp:
            ax.plot(self.sum_regret, label = ["Empirical Regret (UCB)", "Expected Regret (UCB)"])
        else:
            ax.plot(self.sum_regret["Expected"], label = ["Regret (UCB)"])

        ax.set_xlabel('Time')
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(f'Regret ')
        ax.legend()






class HMAB_Results(object):
    # For known variance
    def __init__(self, environment, policy, horizon, r_name):
        self.environment = environment
        self.nbBands = self.environment.nbBands
        self.horizon = horizon
        self.policy = policy
        self.var_flag = policy.estimator.var_flag
        self.name = r_name
        self.time = 0
        self.decision_history = np.zeros([horizon, 3]) # |band_index|bin_index|observation
        self.pre_estimator = deepcopy(policy.estimator)
        self.post_estimator = None
        self.initialize_parameter_history()

        self.dom_results = {"best_band": np.zeros([horizon,1]),
                            "is_dom": np.zeros([horizon,1]),
                            "a_i": np.zeros([horizon, self.nbBands])}




    def initialize_parameter_history(self):
        self.param_strings = self.pre_estimator.parameter_strings
        nbParams= len(self.param_strings)
        self.parameter_history = {}
        for b in range(self.nbBands):
            self.parameter_history[b] = {}
            for param in self.param_strings:
                self.parameter_history[b][param] = []

    def store(self, band_index, bin_index, observations, latest_params):

        if isinstance(observations, (np.ndarray,tuple,list)):
            num_obs = observations.__len__()
            self.decision_history[self.time:self.time + num_obs, 0] = band_index
            self.decision_history[self.time:self.time + num_obs, 1] = bin_index
            self.decision_history[self.time:self.time + num_obs, 2] = observations
            self.extract_params(latest_params)
            self.time += num_obs

    def store_dom_results(self, dom_results):
        self.dom_results["best_band"][self.time] = dom_results[0]
        self.dom_results["is_dom"][self.time] = dom_results[1]
        self.dom_results["a_i"][self.time,:] = dom_results[2]
        if dom_results[1] > 0:
            self.dom_time = self.time
            final_dom = np.concatenate([x[:self.dom_time + 1] for x in self.dom_results.values()], axis=1)
            final_a_i = final_dom[:,2:]
            self.final_dom = pd.DataFrame(final_dom)
            self.final_a_i = pd.DataFrame(final_a_i)


    def extract_params(self, latest_params):
        # Columns correpsond to bands, rows to params, thus we transpose to iterate through the orignal bands
        b = 0
        for band_params in latest_params.T:
            p = 0
            for params in self.pre_estimator.parameter_strings:
                self.parameter_history[b][params].append(band_params[p])
                p += 1
            b += 1

    def store_estimator(self):
        self.post_estimator = self.policy.estimator

    def finalize_results(self):
        print(f"Simulation {self.name} Finished  - creating result DataFrames")
        self.post_estimator = self.policy.estimator
        exp_reward = [self.get_expected_reward(int(band), int(bin)) for band, bin in self.decision_history[:, :2]]
        self.decision_history = np.concatenate([self.decision_history, np.array([exp_reward]).T], axis=1)
        self.decision_history = pd.DataFrame(self.decision_history, columns=["Band", "Bin", "Reward", "Exp Reward"])
        # final_dom = np.concatenate([x for x in self.dom_results.values()], axis = 1)
        # col = ["best_band", "is_dom"].extend([f"Pr(B_max({x}))" for x in range(self.nbBands)])
        # self.final_dom = pd.DataFrame(final_dom, columns = col)
        # self.a_i = pd.DataFrame(self.a_i)
        df_ph = {}

        for key, item in self.parameter_history.items():
            df_ph[key] = pd.DataFrame.from_dict(item)
        self.parameter_history = df_ph
        self.create_summary_table()
        self.compute_regret()

    def store_results(self, file):
        # Start with table
        figs = []

        fig, ax = plt.subplots(figsize=(12,4))
        figs.append(fig)
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=self.summary_df.values, colLabels=self.summary_df.columns, loc='center')
        pp = PdfPages(file)
        #pp.savefig(fig, bbox_inches = "tight")
        plt.rcParams["figure.figsize"] = (12, 8)
        figs.append(self.plot_decision_history())
        plt.rcParams["figure.figsize"] = (12, 4)
        figs.extend(self.plot_bands_param_history(decision_history= True))
        #figs = [plt.figure(n) for n in plt.get_fignums()]

        for fig in figs:

            fig.savefig(pp, format='pdf')
        pp.close()


    def save_and_plot(self, plot = False):
        self.post_estimator = self.policy.estimator

        if plot:
            self.plot_decision_history()
            self.plot_band_params_history()
        self.summarize_results()

    def plot_posterior_param_history(self):
        fig, axes = plt.subplots(self.nbBands,2, sharey= "col")
        if self.nbBands > 1:
            for b in range(self.nbBands):
                c = f"C{b}"
                axes[b,0].plot(self.nu_history[:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon), self.LCI_history[:,b], self.UCI_history[:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.rho_history[:,b], color = c, label = f"B({b})")
            axes[0,0].set_title("Mean of Band Posterior")
            axes[0,1].set_title("Variance of Band Posterior")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.nu_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.rho_history[2:, b], color=c, label=f"B({b})")
            axes[0].set_title("Mean of Band Posterior")
            axes[1].set_title("Variance of Band Posterior")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()

    def plot_bands_param_history(self, decision_history = False, bands = None, params = None):
        figs = []
        self.get_param_max()
        if bands is None:
            bands = range(self.nbBands)
        for b in bands:
            figs.extend(self.plot_band_param_history(decision_history, b))
        return figs


    def plot_band_param_history(self, decision_history = False, band = None):
        if band is None:
            print("NO BAND PASSED TO plot_band_param_history()")
            return
        band_params = self.parameter_history[band]
        figs = []
        if decision_history:
            fig0 = self.plot_band_decision_history(band)
            figs.append(fig0)
        #Plot Band Variance parameters and samples
        ylims = [m + 1 for m in self.param_max.values()]



        fig2, axes2 = plt.subplots(ncols = 3)
        band_params[["rho", "nu", "theta"]].plot.line(subplots = True, ax = axes2, title = ["Variance", "Mean", "Samples"], xlabel = "Round (t)", color = [f"C{b}" for b in range(3,6)])
        fig2.suptitle(f"[{self.name}] Band {band} Mean Posterior Parameters")
        ind= 0
        for ax in axes2:
            ax.set_ylim(0, ylims[ind])
            ind+=1

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig2.tight_layout()
        figs.append(fig2)
        return figs


    def get_param_max(self):
        # Create dictionary of parameter names
        self.param_max = {key : 0 for key in self.param_strings}
        for b in range(self.nbBands):
            for param in self.param_strings:
                self.param_max[param] = max(self.parameter_history[b][param].max(), self.param_max[param])
                if self.param_max[param] is np.nan:
                    self.param_max[param] = 0


    def plot_band_params_history(self):
        fig, axes = plt.subplots(self.nbBands,3, sharey= "col")
        if self.nbBands > 1:
            for b in range(self.nbBands):
                c = f"C{b}"
                axes[b,0].plot(self.nu_history[1:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon-1), self.LCI_history[1:,b], self.UCI_history[1:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.rho_history[1:,b], color = c, label = f"B({b})")
                axes[b,2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0,0].set_title("Mean of Band Posterior")
            axes[0,1].set_title("Variance of Band Posterior")
            axes[0,2].set_title("Estimate of Band Variance Parameter")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.nu_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.rho_history[1:, b], color=c, label=f"B({b})")
            axes[2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0].set_title("Mean of Band Posterior")
            axes[1].set_title("Variance of Band Posterior")
            axes[2].set_title("Estimate of Band Variance Parameter")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()


    def plot_posterior_history(self, band_indices = 0, y_lim = None):
        fig, ax = plt.subplots()
        if not hasattr(band_indices, '__iter__'):
            band_indices = [band_indices]
        for band in band_indices:
            self.post_estimator.bands[band].plot_posterior_history(ax)

        ax.set_title(self.name + " - Posterior History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_mu = 0
            for band in self.environment.bands:
                temp = band.rv.ppf(0.99)
                if temp > max_mu:
                    max_mu = temp
            ax.set_ylim(0, max_mu+1)
        #fig.show()

    def plot_posterior_variance_history(self, band_indices = 0, y_lim = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        if not hasattr(band_indices, '__iter__'):
            band_indices = [band_indices]
        for band in band_indices:
            self.post_estimator.bands[band].plot_posterior_variance_history(ax)

        ax.set_title(self.name + " - Posterior Variance History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_var = 0
            for band in self.environment.bands:
                var = (band.sigma**2 + band.bin_params**2)
                if var > max_var:
                    max_var = var
            ax.set_ylim(0, max_var+1)
        #fig.show()

    def plot_band_decision_history(self, band):

        fig, axes = plt.subplots()

        y = np.zeros([self.decision_history.shape[0]]) - 100
        x = range(self.decision_history.shape[0])
        color = f"C{band}"
        # y = self.decision_history[self.decision_history["Band"] == 0]
        rounds = (self.decision_history["Band"] == band).to_numpy()
        # rounds = np.where(self.decision_history[:, 0] == b)
        y[rounds] = self.decision_history[rounds]["Bin"].to_numpy()
        axes.scatter(x=x, y=y[:], color=color, label=f"B{band}")
        axes.set_ylim(0)
        axes.legend()
        axes.set_title(f"[{self.name}] Band ({band}) Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def plot_decision_history(self):
        nbBands = self.environment.nbBands


        fig, axes = plt.subplots(nbBands,1)

        y= np.zeros([self.decision_history.shape[0], nbBands]) - 10
        x = range(self.decision_history.shape[0])

        for b in range(nbBands):
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Band"] == 0]
            rounds = (self.decision_history["Band"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds,b] = self.decision_history[rounds]["Bin"].to_numpy()
            if nbBands > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"B{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title(f"[{self.name}] Decision History")
                axes[b].set_ylabel("Channel Index")
                axes[b].set_xlabel("Round t")

            else:
                axes.scatter(x=x, y=y[:, b], color=color, label=f"B{b}")
                axes.set_ylim(0)
                axes.legend()
                axes.set_title(f"[{self.name}] Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def plot_sorted_decision_history(self):
        nbBands = self.environment.nbBands
        fig, axes = plt.subplots(nbBands,1)
        y= np.zeros([self.decision_history.shape[0], nbBands]) - 10
        x = range(self.decision_history.shape[0])
        for b in range(nbBands):
            sort_index = self.environment.bands[b].sort_index
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Band"] == 0]
            rounds = (self.decision_history["Band"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds, b] = sort_index[[int(i) for i in self.decision_history[rounds]["Bin"].values]]
            if nbBands > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"B{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title(f"[{self.name}] Decision History - Sorted")
                axes[b].set_ylabel("Sorted Channel Index")
                axes[b].set_xlabel("Round t")

            else:
                axes.scatter(x=x, y=y[:, b], color=color, label=f"B{b}")
                axes.set_ylim(0)
                axes.legend()
                axes.set_title(f"[{self.name}] Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def create_summary_table(self):
        band_index = 0
        d = {}
        pull_count = self.decision_history.value_counts(subset="Band")
        for b in self.post_estimator.bands:
            pulls = pull_count[band_index]
            true_distribution = self.environment.bands[band_index].__sampstr2__()
            mean_posterior = self.post_estimator.bands[band_index].get_mean_posterior_str()
            d[band_index] = [band_index, pulls, true_distribution, mean_posterior]
            band_index += 1
        summary_df = pd.DataFrame.from_dict(d, orient = "index", columns = ['Band','Pulls','True Dist','Mean Post Dist'])
        self.summary_df = summary_df

    def summarize_results(self, bins = False):
        # Number of pulls
        # True value
        # Posterior
        band_index = 0
        d= {}
        pull_count = self.decision_history.value_counts(subset = "Band")
        for b in self.post_estimator.bands:
            pulls = pull_count[band_index]
            true_distribution = self.environment.bands[band_index].__sampstr2__()
            mean_posterior = self.post_estimator.bands[band_index].get_mean_posterior_str()
            d[band_index] = [pulls, true_distribution, mean_posterior]
            band_index += 1
        print ("{:<8} {:<10} {:<20} {:<20}".format('Band','Pulls','True Dist','Mean Post Dist'))
        for k, v in d.items():
            pulls, dist, post = v
            print("{:<8} {:<10} {:<20} {:<20}".format(k, pulls, dist, post))
        if bins:
            print("Best Bins for each band")
            d2 = {}
            for b in range(self.nbBands):
                best_bin = self.environment.best_bins[b]
                mean = self.environment.best_means[b]
                observations = self.post_estimator.bands[b].bins[best_bin].observations
                pulls = len(observations)
                d2[b] = [best_bin, f"{mean:.3f}", pulls]
            print ("{:<8} {:<10} {:<20} {:<20}".format('Band','Bin','True Mean','Pulls'))
            for k, v in d2.items():
                best_bin, mean, pulls = v
                print("{:<8} {:<10} {:<20} {:<20}".format(k, best_bin, mean, pulls))

    def compute_regret(self):
        opt_reward = np.max(self.environment.best_means)
        real_regret = opt_reward - self.decision_history['Reward']
        expected_regret = opt_reward - self.decision_history["Exp Reward"]

        real_regret_sum = real_regret.cumsum()
        expected_regret_sum = expected_regret.cumsum()

        self.regret = pd.concat({"Real": real_regret, "Expected": expected_regret}, axis = 1, )
        self.sum_regret = pd.concat({"Real": real_regret_sum, "Expected": expected_regret_sum,}, axis = 1)

    def get_expected_reward(self, band, bin):
        return self.environment.bin_means[band][bin]

    def plot_cumulative_regret(self, fig = None, ax = None, emp = False):
        if ax is None:
            fig, ax = plt.subplots()
        if emp:
            ax.plot(self.sum_regret, label = ["Empirical Regret (HTS)", "Expected Regret (HTS)"])
        else:
            ax.plot(self.sum_regret['Expected'], label = ["Regret (HTS)"])

        ax.set_xlabel('Time')
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(f'Regret ')
        ax.legend()

    def plot_round_param(self, t):
        fig, axes = plt.subplots(2,1, sharex= True)
        post_rvs = []
        for band_index in range(self.nbBands):
            color = f"C{band_index}"

            rho, nu, theta = self.parameter_history[band_index].iloc[t].values
            true_sigma = self.environment.bands[band_index].sample_sigma
            post_rv = norm(loc = nu, scale = np.sqrt(rho))
            post_rvs.append(post_rv)
            vals = post_rv.ppf([0.01,0.99])
            x = np.linspace(vals[0], vals[1], 100)
            axes[0].plot(x, post_rv.pdf(x), color = color, label = f"Band {band_index} - N({nu:0.2f},{rho:0.2f})")
            axes[0].axvline(theta, color = color, label= f"Band ({band_index} sample)")

            dist_rv = norm(loc = theta, scale = true_sigma)
            vals = dist_rv.ppf([0.01,0.99])
            x = np.linspace(vals[0], vals[1], 100)
            axes[1].plot(x, dist_rv.pdf(x), color = color, label = f"Band {band_index} - N({theta:0.2f}, {true_sigma**2:0.2f})")

        axes[0].legend()
        axes[0].set_title(f"Round {t} Posteriors")
        axes[1].legend()
        axes[1].set_title(f"Round {t} Distribution Estimates")
        return post_rvs

    def compute_error_prob(self, rv_1, rv_2 ):
        # computes the probability rv_2 < rv_1
        mu = rv_2.kwds['loc'] - rv_1.kwds['loc']
        sigma = np.sqrt(rv_2.kwds['scale']**2 + rv_1.kwds['scale']**2)
        rv = norm(loc = mu, scale = sigma)
        error_prob = rv.sf(0)
        return error_prob

    def simulate_error_prob(self, posteriors, samples = 10000):
        draws = np.zeros([samples, len(posteriors)])
        i = 0
        for rv in posteriors:
            draws[:,i] = rv.rvs(size = samples).T
            i+=1
        chosen = pd.DataFrame(np.argmax(draws, axis=1))
        chosen_freq = chosen.value_counts(normalize= True).to_dict()
        return chosen_freq

    def plot_stoch_dom(self):
        ax = self.final_a_i.plot.bar(stacked = True)
        ax.legend(title = 'Band')
        ax.set_ylabel("Pr(Max X_i)")
        ax.set_xlabel("Round (t)")
        plt.get_current_fig_manager().window.state('zoomed')
        plt.tight_layout()
        return ax


class Result(object):

    def __init__(self, environment, policy, horizon, r_name):
        self.environment = environment
        self.nbBands = self.environment.nbBands
        self.horizon = horizon
        self.policy = policy
        self.var_flag = policy.estimator.var_flag
        self.name = r_name
        self.time = 0
        self.decision_history = np.zeros([horizon, 3])
        self.pre_estimator = deepcopy(policy.estimator)
        self.post_estimator = None
        self.initialize_parameter_history()



    def initialize_parameter_history(self):
        self.param_strings = self.pre_estimator.parameter_strings
        nbParams= len(self.param_strings)
        self.parameter_history = {}
        for b in range(self.nbBands):
            self.parameter_history[b] = {}
            for param in self.param_strings:
                self.parameter_history[b][param] = []

    def store(self, band_index, bin_index, observations, latest_params):

        if isinstance(observations, (np.ndarray,tuple,list)):
            num_obs = observations.__len__()
            self.decision_history[self.time:self.time + num_obs, 0] = band_index
            self.decision_history[self.time:self.time + num_obs, 1] = bin_index
            self.decision_history[self.time:self.time + num_obs, 2] = observations
            self.extract_params(latest_params)
            self.time += num_obs

    def extract_params(self, latest_params):
        # Columns correpsond to bands, rows to params, thus we transpose to iterate through the orignal bands
        b = 0
        for band_params in latest_params.T:
            p = 0
            for params in self.pre_estimator.parameter_strings:
                self.parameter_history[b][params].append(band_params[p])
                p += 1
            b += 1

    def store_estimator(self):
        self.post_estimator = self.policy.estimator

    def finalize_results(self):
        print(f"Simulation {self.name} Finished  - creating result DataFrames")
        self.post_estimator = self.policy.estimator
        self.decision_history = pd.DataFrame(self.decision_history, columns=["Band", "Bin", "Reward"])
        df_ph = {}
        for key, item in self.parameter_history.items():
            df_ph[key] = pd.DataFrame.from_dict(item)
        self.parameter_history = df_ph
        self.create_summary_table()

    def store_results(self, file):
        # Start with table
        figs = []

        fig, ax = plt.subplots(figsize=(12,4))
        figs.append(fig)
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=self.summary_df.values, colLabels=self.summary_df.columns, loc='center')
        pp = PdfPages(file)
        #pp.savefig(fig, bbox_inches = "tight")
        plt.rcParams["figure.figsize"] = (12, 8)
        figs.append(self.plot_decision_history())
        plt.rcParams["figure.figsize"] = (12, 4)
        figs.extend(self.plot_bands_param_history(decision_history= True))
        #figs = [plt.figure(n) for n in plt.get_fignums()]

        for fig in figs:

            fig.savefig(pp, format='pdf')
        pp.close()


    def save_and_plot(self, plot = False):
        self.post_estimator = self.policy.estimator

        if plot:
            self.plot_decision_history()
            self.plot_band_params_history()
        self.summarize_results()

    def plot_posterior_param_history(self):
        fig, axes = plt.subplots(self.nbBands,2, sharey= "col")
        if self.nbBands > 1:
            for b in range(self.nbBands):
                c = f"C{b}"
                axes[b,0].plot(self.nu_history[:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon), self.LCI_history[:,b], self.UCI_history[:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.rho_history[:,b], color = c, label = f"B({b})")
            axes[0,0].set_title("Mean of Band Posterior")
            axes[0,1].set_title("Variance of Band Posterior")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.nu_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.rho_history[2:, b], color=c, label=f"B({b})")
            axes[0].set_title("Mean of Band Posterior")
            axes[1].set_title("Variance of Band Posterior")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()

    def plot_bands_param_history(self, decision_history = False, bands = None, params = None):
        figs = []
        self.get_param_max()
        if bands is None:
            bands = range(self.nbBands)
        for b in bands:
            figs.extend(self.plot_band_param_history(decision_history, b))
        return figs


    def plot_band_param_history(self, decision_history = False, band = None):
        if band is None:
            print("NO BAND PASSED TO plot_band_param_history()")
            return
        band_params = self.parameter_history[band]
        figs = []
        if decision_history:
            fig0 = self.plot_band_decision_history(band)
            figs.append(fig0)
        #Plot Band Variance parameters and samples
        fig1, axes1 = plt.subplots(ncols=3)
        ylims = [m + 1 for m in self.param_max.values()]


        band_params[["eta", "df", "sigma"]].plot.line(subplots = True, ax = axes1, title = ["Scale", "Degrees Freedom", "Samples"], xlabel = "Round (t)", color = [f"C{b}" for b in range(3)])
        fig1.suptitle(f"[{self.name}] Band {band} Variance Posterior Parameters")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig1.tight_layout()
        figs.append(fig1)
        fig2, axes2 = plt.subplots(ncols = 3)
        band_params[["rho", "nu", "theta"]].plot.line(subplots = True, ax = axes2, title = ["Variance", "Mean", "Samples"], xlabel = "Round (t)", color = [f"C{b}" for b in range(3,6)])
        fig2.suptitle(f"[{self.name}] Band {band} Mean Posterior Parameters")
        ind= 0
        for ax in np.concatenate([axes1, axes2]):
            ax.set_ylim(0, ylims[ind])
            ind+=1
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig2.tight_layout()
        figs.append(fig2)
        return figs


    def get_param_max(self):
        # Create dictionary of parameter names
        self.param_max = {key : 0 for key in self.param_strings}
        for b in range(self.nbBands):
            for param in self.param_strings:
                self.param_max[param] = max(self.parameter_history[b][param].max(), self.param_max[param])
                if self.param_max[param] is np.nan:
                    self.param_max[param] = 0





    def plot_band_params_history(self):
        fig, axes = plt.subplots(self.nbBands,3, sharey= "col")
        if self.nbBands > 1:
            for b in range(self.nbBands):
                c = f"C{b}"
                axes[b,0].plot(self.nu_history[1:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon-1), self.LCI_history[1:,b], self.UCI_history[1:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.rho_history[1:,b], color = c, label = f"B({b})")
                axes[b,2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0,0].set_title("Mean of Band Posterior")
            axes[0,1].set_title("Variance of Band Posterior")
            axes[0,2].set_title("Estimate of Band Variance Parameter")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.nu_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.rho_history[1:, b], color=c, label=f"B({b})")
            axes[2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0].set_title("Mean of Band Posterior")
            axes[1].set_title("Variance of Band Posterior")
            axes[2].set_title("Estimate of Band Variance Parameter")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()


    def plot_posterior_history(self, band_indices = 0, y_lim = None):
        fig, ax = plt.subplots()
        if not hasattr(band_indices, '__iter__'):
            band_indices = [band_indices]
        for band in band_indices:
            self.post_estimator.bands[band].plot_posterior_history(ax)

        ax.set_title(self.name + " - Posterior History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_mu = 0
            for band in self.environment.bands:
                temp = band.rv.ppf(0.99)
                if temp > max_mu:
                    max_mu = temp
            ax.set_ylim(0, max_mu+1)
        #fig.show()

    def plot_posterior_variance_history(self, band_indices = 0, y_lim = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        if not hasattr(band_indices, '__iter__'):
            band_indices = [band_indices]
        for band in band_indices:
            self.post_estimator.bands[band].plot_posterior_variance_history(ax)

        ax.set_title(self.name + " - Posterior Variance History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_var = 0
            for band in self.environment.bands:
                var = (band.sigma**2 + band.bin_params**2)
                if var > max_var:
                    max_var = var
            ax.set_ylim(0, max_var+1)
        #fig.show()

    def plot_band_decision_history(self, band):

        fig, axes = plt.subplots()

        y = np.zeros([self.decision_history.shape[0]]) - 100
        x = range(self.decision_history.shape[0])
        color = f"C{band}"
        # y = self.decision_history[self.decision_history["Band"] == 0]
        rounds = (self.decision_history["Band"] == band).to_numpy()
        # rounds = np.where(self.decision_history[:, 0] == b)
        y[rounds] = self.decision_history[rounds]["Bin"].to_numpy()
        axes.scatter(x=x, y=y[:], color=color, label=f"B{band}")
        axes.set_ylim(0)
        axes.legend()
        axes.set_title(f"[{self.name}] Band ({band}) Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig
    def plot_decision_history(self):
        nbBands = self.environment.nbBands


        fig, axes = plt.subplots(nbBands,1)

        y= np.zeros([self.decision_history.shape[0], nbBands]) - 10
        x = range(self.decision_history.shape[0])

        for b in range(nbBands):
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Band"] == 0]
            rounds = (self.decision_history["Band"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds,b] = self.decision_history[rounds]["Bin"].to_numpy()
            if nbBands > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"B{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title(f"[{self.name}] Decision History")

            else:
                axes.scatter(x=x, y=y[:, b], color=color, label=f"B{b}")
                axes.set_ylim(0)
                axes.legend()
                axes.set_title(f"[{self.name}] Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def create_summary_table(self):
        band_index = 0
        d = {}
        pull_count = self.decision_history.value_counts(subset="Band")
        for b in self.post_estimator.bands:
            pulls = pull_count[band_index]
            true_distribution = self.environment.bands[band_index].__sampstr2__()
            mean_posterior = self.post_estimator.bands[band_index].get_mean_posterior_str()
            variance_posterior = self.post_estimator.bands[band_index].get_variance_posterior_str()
            d[band_index] = [band_index, pulls, true_distribution, mean_posterior, variance_posterior]
            band_index += 1
        summary_df = pd.DataFrame.from_dict(d, orient = "index", columns = ['Band','Pulls','True Dist','Mean Post Dist', 'Var Post Dist'])
        self.summary_df = summary_df

    def summarize_results(self, bins = False):
        # Number of pulls
        # True value
        # Posterior
        band_index = 0
        d= {}
        pull_count = self.decision_history.value_counts(subset = "Band")
        for b in self.post_estimator.bands:
            pulls = pull_count[band_index]
            true_distribution = self.environment.bands[band_index].__sampstr2__()
            mean_posterior = self.post_estimator.bands[band_index].get_mean_posterior_str()
            variance_posterior = self.post_estimator.bands[band_index].get_variance_posterior_str()
            d[band_index] = [pulls, true_distribution, mean_posterior, variance_posterior]
            band_index += 1
        print ("{:<8} {:<10} {:<20} {:<20} {:<20}".format('Band','Pulls','True Dist','Mean Post Dist', 'Var Post Dist'))
        for k, v in d.items():
            pulls, dist, post, sigma_i = v
            print("{:<8} {:<10} {:<20} {:<20} {:<20}".format(k, pulls, dist, post, sigma_i))
        if bins:
            print("Best Bins for each band")
            d2 = {}
            for b in range(self.nbBands):
                best_bin = self.environment.best_bins[b]
                mean = self.environment.best_means[b]
                observations = self.post_estimator.bands[b].bins[best_bin].observations
                pulls = len(observations)
                d2[b] = [best_bin, f"{mean:.3f}", pulls]
            print ("{:<8} {:<10} {:<20} {:<20}".format('Band','Bin','True Mean','Pulls'))
            for k, v in d2.items():
                best_bin, mean, pulls = v
                print("{:<8} {:<10} {:<20} {:<20}".format(k, best_bin, mean, pulls))


class OGResult(object):

    def __init__(self, environment, policy, horizon, r_name):
        self.environment = environment
        self.nbBands = self.environment.nbBands
        self.horizon = horizon
        self.policy = policy
        self.name = r_name
        self.time = 0
        self.decision_history = np.zeros([horizon, 3])
        self.pre_estimator = deepcopy(policy.estimator)
        self.post_estimator = None
        self.nu_history = np.zeros((horizon, self.nbBands)) # History of mean parameter of posterior on band mean
        self.rho_history = np.zeros((horizon, self.nbBands)) # History of variance parameter of posterior on band mean
        self.df_history = np.zeros((horizon, self.nbBands))
        self.eta_history = np.zeros((horizon, self.nbBands))
        self.UCI_history = np.zeros((horizon, self.nbBands))
        self.LCI_history = np.zeros((horizon,self.nbBands))


    def store(self, band_index, bin_index, observations, latest_params):

        if isinstance(observations, (np.ndarray,tuple,list)):
            num_obs = observations.__len__()
            self.decision_history[self.time:self.time + num_obs, 0] = band_index
            self.decision_history[self.time:self.time + num_obs, 1] = bin_index
            self.decision_history[self.time:self.time + num_obs, 2] = observations
            self.nu_history[self.time, :] = latest_params[0]
            self.rho_history[self.time, :] = latest_params[1]
            self.LCI_history[self.time,:] = latest_params[2]
            self.UCI_history[self.time,:] = latest_params[3]
            self.df_history[self.time, :] = latest_params[4]
            self.eta_history[self.time, :] = latest_params[5]
            self.time += num_obs
        else:
            self.decision_history[self.time][0] = band_index
            self.decision_history[self.time][1] = bin_index
            self.decision_history[self.time][2] = observations
            self.nu_history[self.time,:] = latest_params[0]
            self.rho_history[self.time,:] = latest_params[1]
            self.LCI_history[self.time, :] = latest_params[2]
            self.UCI_history[self.time, :] = latest_params[3]

            self.time +=1

    def store_estimator(self):
        self.post_estimator = self.policy.estimator

    def finalize_results(self):
        print(f"Simulation {self.name} Finished  - creating result DataFrames")
        self.post_estimator = self.policy.estimator
        self.decision_history = pd.DataFrame(self.decision_history, columns=["Band", "Bin", "Reward"])
        self.parameter_history = []
        for b in range(self.nbBands):
            self.parameter_history.append(pd.DataFrame([self.nu_history[:,b], self.rho_history[:,b], self.df_history[:,b], self.eta_history[:,b]], columns= ["nu", "rho", "df", "eta"]))

    def save_and_plot(self, plot = False):
        self.post_estimator = self.policy.estimator

        if plot:
            self.plot_decision_history()
            self.plot_band_params_history()
        self.summarize_results()

    def plot_posterior_param_history(self):
        fig, axes = plt.subplots(self.nbBands,2, sharey= "col")
        if self.nbBands > 1:
            for b in range(self.nbBands):
                c = f"C{b}"
                axes[b,0].plot(self.nu_history[:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon), self.LCI_history[:,b], self.UCI_history[:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.rho_history[:,b], color = c, label = f"B({b})")
            axes[0,0].set_title("Mean of Band Posterior")
            axes[0,1].set_title("Variance of Band Posterior")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.nu_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.rho_history[2:, b], color=c, label=f"B({b})")
            axes[0].set_title("Mean of Band Posterior")
            axes[1].set_title("Variance of Band Posterior")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()

    def plot_band_params_history(self):
        fig, axes = plt.subplots(self.nbBands,3, sharey= "col")
        if self.nbBands > 1:
            for b in range(self.nbBands):
                c = f"C{b}"
                axes[b,0].plot(self.nu_history[1:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon-1), self.LCI_history[1:,b], self.UCI_history[1:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.rho_history[1:,b], color = c, label = f"B({b})")
                axes[b,2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0,0].set_title("Mean of Band Posterior")
            axes[0,1].set_title("Variance of Band Posterior")
            axes[0,2].set_title("Estimate of Band Variance Parameter")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.nu_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.rho_history[1:, b], color=c, label=f"B({b})")
            axes[2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0].set_title("Mean of Band Posterior")
            axes[1].set_title("Variance of Band Posterior")
            axes[2].set_title("Estimate of Band Variance Parameter")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()


    def plot_posterior_history(self, band_indices = 0, y_lim = None):
        fig, ax = plt.subplots()
        if not hasattr(band_indices, '__iter__'):
            band_indices = [band_indices]
        for band in band_indices:
            self.post_estimator.bands[band].plot_posterior_history(ax)

        ax.set_title(self.name + " - Posterior History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_mu = 0
            for band in self.environment.bands:
                temp = band.rv.ppf(0.99)
                if temp > max_mu:
                    max_mu = temp
            ax.set_ylim(0, max_mu+1)
        #fig.show()

    def plot_posterior_variance_history(self, band_indices = 0, y_lim = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        if not hasattr(band_indices, '__iter__'):
            band_indices = [band_indices]
        for band in band_indices:
            self.post_estimator.bands[band].plot_posterior_variance_history(ax)

        ax.set_title(self.name + " - Posterior Variance History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_var = 0
            for band in self.environment.bands:
                var = (band.sigma**2 + band.bin_params**2)
                if var > max_var:
                    max_var = var
            ax.set_ylim(0, max_var+1)
        #fig.show()


    def plot_decision_history(self):
        nbBands = self.environment.nbBands


        fig, axes = plt.subplots(nbBands,1)

        y= np.zeros([self.decision_history.shape[0], nbBands]) - 10
        x = range(self.decision_history.shape[0])

        for b in range(nbBands):
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Band"] == 0]
            rounds = (self.decision_history["Band"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds,b] = self.decision_history[rounds]["Bin"].to_numpy()
            if nbBands > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"B{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title("Decision History")

            else:
                axes.scatter(x=x, y=y[:, b], color=color, label=f"B{b}")
                axes.set_ylim(0)
                axes.legend()
                axes.set_title("Decision History")


    def print_final_posterior(self):
        print('*'*20)
        print("Printing the final posteriors for each band")
        for b in self.post_estimator.bands:
            print(b.get_posterior_str())

    def summarize_results(self, bins = False):
        # Number of pulls
        # True value
        # Posterior
        band_index = 0
        d= {}
        pull_count = self.decision_history.value_counts(subset = "Band")
        for b in self.post_estimator.bands:
            pulls = pull_count[band_index]
            true_nu = self.environment.bands[band_index].__sampstr2__()
            post_string = self.post_estimator.bands[band_index].posterior_str()
            sigma_i = self.post_estimator.bands[band_index].get_band_variance_est()
            d[band_index] = [pulls, true_nu, post_string, sigma_i]
            band_index += 1
        print ("{:<8} {:<10} {:<20} {:<20} {:<20}".format('Band','Pulls','True Dist','Theta Post Dist', 'Sigma_i^2 Est'))
        for k, v in d.items():
            pulls, dist, post, sigma_i = v
            print("{:<8} {:<10} {:<20} {:<20} {:<20}".format(k, pulls, dist, post, sigma_i))
        if bins:
            print("Best Bins for each band")
            d2 = {}
            for b in range(self.nbBands):
                best_bin = self.environment.best_bins[b]
                mean = self.environment.best_means[b]
                observations = self.post_estimator.bands[b].bins[best_bin].observations
                pulls = len(observations)
                d2[b] = [best_bin, f"{mean:.3f}", pulls]
            print ("{:<8} {:<10} {:<20} {:<20}".format('Band','Bin','True Mean','Pulls'))
            for k, v in d2.items():
                best_bin, mean, pulls = v
                print("{:<8} {:<10} {:<20} {:<20}".format(k, best_bin, mean, pulls))









