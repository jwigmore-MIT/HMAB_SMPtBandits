import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from BHB.helper import *

# mng = plt.get_current_fig_manager()
# mng.window.state('zoomed')

import pandas as pd


class UCB_Results(object):

    def __init__(self, environment, policy, horizon, r_name):
        self.environment = environment
        self.nbClusters = self.environment.nbClusters
        self.nbArms = self.environment.nbArms
        self.horizon = horizon
        self.policy = policy
        self.name = r_name
        self.decision_history= np.zeros([horizon, 3])
        self.time = 0
        self.pre_estimator = deepcopy(policy.estimator)
        self.post_estimator = None
        self.latest_params = None # NOT IMPLEMENTED

    def store(self, cluster_index, arm_index, observations, latest_params= None):
        if isinstance(observations, (np.ndarray, tuple, list)):
            num_obs = observations.__len__()
            self.decision_history[self.time:self.time + num_obs, 0] = cluster_index
            self.decision_history[self.time:self.time + num_obs, 1] = arm_index
            self.decision_history[self.time:self.time + num_obs, 2] = observations
            self.time += num_obs

    def plot_decision_history(self):
        nbClusters = self.environment.nbClusters
        fig, axes = plt.subplots(nbClusters,1)
        y= np.zeros([self.decision_history.shape[0], nbClusters]) - 10
        x = range(self.decision_history.shape[0])
        for b in range(nbClusters):
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Cluster"] == 0]
            rounds = (self.decision_history["Cluster"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds,b] = self.decision_history[rounds]["Arm"].to_numpy()
            if nbClusters > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"B{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title(f"{self.policy.__str__()} - Decision History")
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
        nbClusters = self.environment.nbClusters
        fig, axes = plt.subplots(nbClusters,1)
        y= np.zeros([self.decision_history.shape[0], nbClusters]) - 10
        x = range(self.decision_history.shape[0])
        for b in range(nbClusters):
            sort_index = self.environment.clusters[b].sort_index
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Cluster"] == 0]
            rounds = (self.decision_history["Cluster"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds, b] = sort_index[[int(i) for i in self.decision_history[rounds]["Arm"].values]]
            if nbClusters > 1:
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
        exp_reward = [self.get_expected_reward(int(cluster), int(arm)) for cluster, arm in self.decision_history[:,:2]]
        self.decision_history = np.concatenate([self.decision_history,np.array([exp_reward]).T], axis=1)
        self.decision_history = pd.DataFrame(self.decision_history, columns=["Cluster", "Arm", "Reward", "Exp Reward"])
        self.post_estimator = self.policy.estimator
        self.compute_regret()

    def compute_regret(self):
        opt_reward = np.max(self.environment.best_means)
        real_regret = opt_reward - self.decision_history['Reward']
        expected_regret = opt_reward - self.decision_history["Exp Reward"]

        real_regret_sum = real_regret.cumsum()
        expected_regret_sum = expected_regret.cumsum()

        self.regret = pd.concat({"Real": real_regret, "Expected": expected_regret}, axis = 1, )
        self.sum_regret = pd.concat({"Real": real_regret_sum, "Expected": expected_regret_sum,}, axis = 1)


    def get_expected_reward(self, cluster, arm):
        return self.environment.arm_means[cluster][arm]

    def plot_cumulative_regret(self, fig = None, ax = None, emp = False):
        if ax is None:
            fig, ax = plt.subplots()
        if emp:
            ax.plot(self.sum_regret, label = [f"Empirical Regret -  {self.policy.__str__()}", "Expected Regret (UCB)"])
        else:
            ax.plot(self.sum_regret["Expected"], label = [f"{self.policy.__str__()}"])

        ax.set_xlabel('Time')
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(f'Regret ')
        ax.legend()

    def summarize_results(self, arms = False):
        # Number of pulls
        # True value
        # Posterior
        cluster_index = 0
        d= {}
        pull_count = self.decision_history.value_counts(subset = "Cluster")
        for b in range(self.nbClusters):
            pulls = pull_count[cluster_index]
            d[cluster_index] = [pulls]
            cluster_index += 1
        print('*'*30)
        print(f'{self.policy.__str__()} - Summary')
        print(f'Cumulative Regret - {self.sum_regret["Expected"].iloc[-1]}')
        print(f'Time-Average Regret - {self.sum_regret["Expected"].iloc[-1]/self.time}')
        print(f'Run time - {self.run_time}')
        print ("{:<8} {:<10} {:<20} {:<20}".format('Cluster','Pulls','True Dist','Mean Post Dist'))
        for k, v in d.items():
            pulls = v
            print("{:<8} {:<10}".format(k, pulls[-1]))
        if arms:
            print("Best Arms for each cluster")
            d2 = {}
            for b in range(self.nbClusters):
                best_arm = self.environment.best_arms[b]
                mean = self.environment.best_means[b]
                pulls = self.post_estimator.counts[get_1D_arm_index(self, b, best_arm)]
                d2[b] = [best_arm, f"{mean:.3f}", pulls]
            print ("{:<8} {:<10} {:<20} {:<20}".format('Cluster','Arm','True Mean','Pulls'))
            for k, v in d2.items():
                best_arm, mean, pulls = v
                print("{:<8} {:<10} {:<20} {:<20}".format(k, best_arm, mean, pulls))



class HMAB_Results(object):
    # For known variance
    def __init__(self, environment, policy, horizon, r_name):
        self.environment = environment
        self.nbClusters = self.environment.nbClusters
        self.horizon = horizon
        self.policy = policy
        self.name = r_name
        self.time = 0
        self.decision_history = np.zeros([horizon, 3]) # |cluster_index|arm_index|observation
        self.pre_estimator = deepcopy(policy.estimator)
        self.post_estimator = None
        self.initialize_parameter_history()
        self.run_time = None

        self.dom_results = {"best_cluster": np.zeros([horizon,1]),
                            "is_dom": np.zeros([horizon,1]),
                            "a_i": np.zeros([horizon, self.nbClusters])}

    def initialize_parameter_history(self):
        self.param_strings = self.pre_estimator.parameter_strings
        nbParams= len(self.param_strings)
        self.parameter_history = {}
        for b in range(self.nbClusters):
            self.parameter_history[b] = {}
            for param in self.param_strings:
                self.parameter_history[b][param] = []

    def store(self, cluster_index, arm_index, observations, latest_params):

        if isinstance(observations, (np.ndarray,tuple,list)):
            num_obs = observations.__len__()
            self.decision_history[self.time:self.time + num_obs, 0] = cluster_index
            self.decision_history[self.time:self.time + num_obs, 1] = arm_index
            self.decision_history[self.time:self.time + num_obs, 2] = observations
            self.extract_params(latest_params)
            self.time += num_obs

    def store_dom_results(self, dom_results):
        self.dom_results["best_cluster"][self.time] = dom_results[0]
        self.dom_results["is_dom"][self.time] = dom_results[1]
        self.dom_results["a_i"][self.time,:] = dom_results[2]
        if dom_results[1] > 0:
            self.dom_time = self.time
            final_dom = np.concatenate([x[:self.dom_time + 1] for x in self.dom_results.values()], axis=1)
            final_a_i = final_dom[:,2:]
            self.final_dom = pd.DataFrame(final_dom)
            self.final_a_i = pd.DataFrame(final_a_i)


    def extract_params(self, latest_params):
        # Columns correpsond to clusters, rows to params, thus we transpose to iterate through the orignal clusters
        b = 0
        for cluster_params in latest_params.T:
            p = 0
            for params in self.pre_estimator.parameter_strings:
                self.parameter_history[b][params].append(cluster_params[p])
                p += 1
            b += 1

    def store_estimator(self):
        self.post_estimator = self.policy.estimator

    def finalize_results(self):
        print(f"Simulation {self.name} Finished  - creating result DataFrames")
        self.post_estimator = self.policy.estimator
        exp_reward = [self.get_expected_reward(int(cluster), int(arm)) for cluster, arm in self.decision_history[:, :2]]
        self.decision_history = np.concatenate([self.decision_history, np.array([exp_reward]).T], axis=1)
        self.decision_history = pd.DataFrame(self.decision_history, columns=["Cluster", "Arm", "Reward", "Exp Reward"])
        # final_dom = np.concatenate([x for x in self.dom_results.values()], axis = 1)
        # col = ["best_cluster", "is_dom"].extend([f"Pr(B_max({x}))" for x in range(self.nbClusters)])
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
        figs.extend(self.plot_clusters_param_history(decision_history= True))
        #figs = [plt.figure(n) for n in plt.get_fignums()]

        for fig in figs:

            fig.savefig(pp, format='pdf')
        pp.close()

    def save_and_plot(self, plot = False):
        self.post_estimator = self.policy.estimator

        if plot:
            self.plot_decision_history()
            self.plot_cluster_params_history()
        self.summarize_results()

    def plot_posterior_param_history(self):
        fig, axes = plt.subplots(self.nbClusters,2, sharey= "col")
        if self.nbClusters > 1:
            for b in range(self.nbClusters):
                c = f"C{b}"
                axes[b,0].plot(self.u_i_history[:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon), self.LCI_history[:,b], self.UCI_history[:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.v_i_history[:,b], color = c, label = f"B({b})")
            axes[0,0].set_title("Mean of Cluster Posterior")
            axes[0,1].set_title("Variance of Cluster Posterior")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.u_i_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.v_i_history[2:, b], color=c, label=f"B({b})")
            axes[0].set_title("Mean of Cluster Posterior")
            axes[1].set_title("Variance of Cluster Posterior")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()

    def plot_clusters_param_history(self, decision_history = False, clusters = None, params = None):
        figs = []
        self.get_param_max()
        if clusters is None:
            clusters = range(self.nbClusters)
        for b in clusters:
            figs.extend(self.plot_cluster_param_history(decision_history, b))
        return figs


    def plot_cluster_param_history(self, decision_history = False, cluster = None):
        if cluster is None:
            print("NO BAND PASSED TO plot_cluster_param_history()")
            return
        cluster_params = self.parameter_history[cluster]
        figs = []
        if decision_history:
            fig0 = self.plot_cluster_decision_history(cluster)
            figs.append(fig0)
        #Plot Cluster Variance parameters and samples
        ylims = [m + 1 for m in self.param_max.values()]



        fig2, axes2 = plt.subplots(ncols = 3)
        cluster_params[["v_i", "u_i", "theta_samples"]].plot.line(subplots = True, ax = axes2, title = ["Posterior Variance", "Posterior Mean", "Samples"], xlabel = "Round (t)", color = [f"C{b}" for b in range(3,6)])
        fig2.suptitle(f"{self.policy.__str__()} -  Cluster {cluster} Mean Posterior Parameters")
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
        for b in range(self.nbClusters):
            for param in self.param_strings:
                self.param_max[param] = max(self.parameter_history[b][param].max(), self.param_max[param])
                if self.param_max[param] is np.nan:
                    self.param_max[param] = 0


    def plot_cluster_params_history(self):
        '''
        Plots the clusters' means parameter's posterior mean, variance, and samples
        :return:
        '''
        fig, axes = plt.subplots(self.nbClusters,3, sharey= "col")
        if self.nbClusters > 1:
            for b in range(self.nbClusters):
                c = f"C{b}"
                axes[b,0].plot(self.u_i_history[1:,b], color = c, label = f"B({b})")
                axes[b,0].fill_between(range(self.horizon-1), self.LCI_history[1:,b], self.UCI_history[1:,b], color = c, alpha=0.3, label="95% Credible Interval")
                axes[b,1].plot(self.v_i_history[1:,b], color = c, label = f"B({b})")
                axes[b,2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0,0].set_title("Mean of Cluster Posterior")
            axes[0,1].set_title("Variance of Cluster Posterior")
            axes[0,2].set_title("Estimate of Cluster Variance Parameter")
        else:
            c = f"C1"
            b = 0
            axes[0].plot(self.u_i_history[1:, b], color=c, label=f"B({b})")
            axes[0].fill_between(range(self.horizon-1), self.LCI_history[1:, b], self.UCI_history[1:, b], color=c, alpha=0.3, label="95% Credible Interval")
            axes[1].plot(self.v_i_history[1:, b], color=c, label=f"B({b})")
            axes[2].plot(self.eta_history[1:, b], color = c, label=f"B({b})")
            axes[0].set_title("Mean of Cluster Posterior")
            axes[1].set_title("Variance of Cluster Posterior")
            axes[2].set_title("Estimate of Cluster Variance Parameter")
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig.tight_layout()


    def plot_posterior_history(self, cluster_indices = 0, y_lim = None):
        fig, ax = plt.subplots()
        if not hasattr(cluster_indices, '__iter__'):
            cluster_indices = [cluster_indices]
        for cluster in cluster_indices:
            self.post_estimator.cluster_beliefs[cluster].plot_posterior_history(ax)

        ax.set_title(self.name + " - Posterior History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_mu = 0
            for cluster in self.environment.clusters:
                temp = cluster.rv.ppf(0.99)
                if temp > max_mu:
                    max_mu = temp
            ax.set_ylim(0, max_mu+1)
        #fig.show()

    def plot_posterior_variance_history(self, cluster_indices = 0, y_lim = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        if not hasattr(cluster_indices, '__iter__'):
            cluster_indices = [cluster_indices]
        for cluster in cluster_indices:
            self.post_estimator.cluster_beliefs[cluster].plot_posterior_variance_history(ax)

        ax.set_title(self.name + " - Posterior Variance History")
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            max_var = 0
            for cluster in self.environment.clusters:
                var = (cluster.sigma**2 + cluster.arm_params**2)
                if var > max_var:
                    max_var = var
            ax.set_ylim(0, max_var+1)
        #fig.show()

    def plot_cluster_decision_history(self, cluster):

        fig, axes = plt.subplots()

        y = np.zeros([self.decision_history.shape[0]]) - 100
        x = range(self.decision_history.shape[0])
        color = f"C{cluster}"
        # y = self.decision_history[self.decision_history["Cluster"] == 0]
        rounds = (self.decision_history["Cluster"] == cluster).to_numpy()
        # rounds = np.where(self.decision_history[:, 0] == b)
        y[rounds] = self.decision_history[rounds]["Arm"].to_numpy()
        axes.scatter(x=x, y=y[:], color=color, label=f"B{cluster}")
        axes.set_ylim(0)
        axes.legend()
        axes.set_title(f"[{self.name}] Cluster ({cluster}) Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def plot_decision_history(self):
        nbClusters = self.environment.nbClusters


        fig, axes = plt.subplots(nbClusters,1)

        y= np.zeros([self.decision_history.shape[0], nbClusters]) - 10
        x = range(self.decision_history.shape[0])

        for b in range(nbClusters):
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Cluster"] == 0]
            rounds = (self.decision_history["Cluster"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds,b] = self.decision_history[rounds]["Arm"].to_numpy()
            if nbClusters > 1:
                axes[b].scatter(x = x ,y = y[:,b], color = color, label = f"C{b}")
                axes[b].set_ylim(0)
                axes[b].legend()
                axes[0].set_title(f"{self.policy.__str__()} -  Decision History")
                axes[b].set_ylabel("Channel Index")
                axes[b].set_xlabel("Round t")

            else:
                axes.scatter(x=x, y=y[:, b], color=color, label=f"B{b}")
                axes.set_ylim(0)
                axes.legend()
                axes.set_title(f"{self.policy.__str__()} - Decision History")
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.tight_layout()
        return fig

    def plot_sorted_decision_history(self):
        nbClusters = self.environment.nbClusters
        fig, axes = plt.subplots(nbClusters,1)
        y= np.zeros([self.decision_history.shape[0], nbClusters]) - 10
        x = range(self.decision_history.shape[0])
        for b in range(nbClusters):
            sort_index = self.environment.clusters[b].sort_index
            color = f"C{b}"
            #y = self.decision_history[self.decision_history["Cluster"] == 0]
            rounds = (self.decision_history["Cluster"] == b).to_numpy()
            #rounds = np.where(self.decision_history[:, 0] == b)
            y[rounds, b] = sort_index[[int(i) for i in self.decision_history[rounds]["Arm"].values]]
            if nbClusters > 1:
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
        cluster_index = 0
        d = {}
        pull_count = self.decision_history.value_counts(subset="Cluster")
        for b in self.post_estimator.cluster_beliefs:
            try:
                pulls = pull_count[cluster_index]
            except KeyError:
                pulls = 0
            true_distribution = self.environment.clusters[cluster_index].__sampstr2__()
            mean_posterior = self.post_estimator.cluster_beliefs[cluster_index].get_cluster_belief_str()
            d[cluster_index] = [cluster_index, pulls, true_distribution, mean_posterior]
            cluster_index += 1
        summary_df = pd.DataFrame.from_dict(d, orient = "index", columns = ['Cluster','Pulls','True Dist','Mean Post Dist'])
        self.summary_df = summary_df

    def summarize_results(self, arms = False):
        # Number of pulls
        # True value
        # Posterior
        cluster_index = 0
        d= {}
        pull_count = self.decision_history.value_counts(subset = "Cluster")
        for b in self.post_estimator.cluster_beliefs:
            try:
                pulls = pull_count[cluster_index]
            except KeyError:
                pulls = 0
            true_distribution = self.environment.clusters[cluster_index].__sampstr2__()
            mean_posterior = self.post_estimator.cluster_beliefs[cluster_index].get_cluster_belief_str()
            d[cluster_index] = [pulls, true_distribution, mean_posterior]
            cluster_index += 1
        print('*'*30)
        print(f'{self.policy.__str__()} - Summary')
        print(f'Cumulative Regret - {self.sum_regret["Expected"].iloc[-1]}')
        print(f'Time-Average Regret - {self.sum_regret["Expected"].iloc[-1]/self.time}')
        print(f'Run time - {self.run_time}')
        print ("{:<8} {:<10} {:<20} {:<20}".format('Cluster','Pulls','True Dist','Mean Post Dist'))
        for k, v in d.items():
            pulls, dist, post = v
            print("{:<8} {:<10} {:<20} {:<20}".format(k, pulls, dist, post))
        if arms:
            print("Best Arms for each cluster")
            d2 = {}
            for b in range(self.nbClusters):
                best_arm = self.environment.best_arms[b]
                mean = self.environment.best_means[b]
                observations = self.post_estimator.cluster_beliefs[b].arm_beliefs[best_arm].observations
                pulls = len(observations)
                d2[b] = [best_arm, f"{mean:.3f}", pulls]
            print ("{:<8} {:<10} {:<20} {:<20}".format('Cluster','Arm','True Mean','Pulls'))
            for k, v in d2.items():
                best_arm, mean, pulls = v
                print("{:<8} {:<10} {:<20} {:<20}".format(k, best_arm, mean, pulls))

    def compute_regret(self):
        opt_reward = np.max(self.environment.best_means)
        real_regret = opt_reward - self.decision_history['Reward']
        expected_regret = opt_reward - self.decision_history["Exp Reward"]

        real_regret_sum = real_regret.cumsum()
        expected_regret_sum = expected_regret.cumsum()

        self.regret = pd.concat({"Real": real_regret, "Expected": expected_regret}, axis = 1, )
        self.sum_regret = pd.concat({"Real": real_regret_sum, "Expected": expected_regret_sum,}, axis = 1)

    def get_expected_reward(self, cluster, arm):
        return self.environment.arm_means[cluster][arm]

    def plot_cumulative_regret(self, fig = None, ax = None, emp = False):
        if ax is None:
            fig, ax = plt.subplots()
        if emp:
            ax.plot(self.sum_regret, label = [f"Empirical Regret - {self.policy.__str__()}", "Expected Regret (HTS)"])
        else:
            ax.plot(self.sum_regret['Expected'], label = [f"{self.policy.__str__()}"])

        ax.set_xlabel('Time')
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(f'Regret ')
        ax.legend()

    def plot_round_param(self, t):
        fig, axes = plt.subplots(2,1, sharex= True)
        post_rvs = []
        for cluster_index in range(self.nbClusters):
            color = f"C{cluster_index}"

            v_i, u_i, theta_samples = self.parameter_history[cluster_index].iloc[t].values
            true_sigma = self.environment.clusters[cluster_index].sample_sigma
            post_rv = norm(loc = u_i, scale = np.sqrt(v_i))
            post_rvs.append(post_rv)
            vals = post_rv.ppf([0.01,0.99])
            x = np.linspace(vals[0], vals[1], 100)
            axes[0].plot(x, post_rv.pdf(x), color = color, label = f"Cluster {cluster_index} - N({u_i:0.2f},{v_i:0.2f})")
            axes[0].axvline(theta_samples, color = color, label= f"Cluster ({cluster_index} sample)")

            dist_rv = norm(loc = theta_samples, scale = true_sigma)
            vals = dist_rv.ppf([0.01,0.99])
            x = np.linspace(vals[0], vals[1], 100)
            axes[1].plot(x, dist_rv.pdf(x), color = color, label = f"Cluster {cluster_index} - N({theta_samples:0.2f}, {true_sigma**2:0.2f})")

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
        ax.legend(title = 'Cluster')
        ax.set_ylabel("Pr(Max X_i)")
        ax.set_xlabel("Round (t)")
        plt.get_current_fig_manager().window.state('zoomed')
        plt.tight_layout()
        return ax















