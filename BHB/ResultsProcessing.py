import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from BHB.MultiSimulationDriver import MultiSimulationDriver


class SimHelper(object):

    def __init__(self, driver):
        self.driver = driver
        self.pol_regrets = None
        self.avg_regret_df = None

    def plot_all_envs(self):
        for trial in self.driver.trials.values():
            env = trial["env"]
            env.plot_clusters_dists(nrm = False, priors = True, gen = False, samp = True)

    def plot_envs(self, env_ids, title = None):
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        for env_id in env_ids:
            env = self.driver.trials[env_id]["env"]
            env.plot_clusters_dists(nrm=False, priors=True, gen=False, samp=True, title = title)

    def plot_environment_paper(self, env_id, title = None, xlim = None):
        env = self.driver.trials[env_id]["env"]
        env.plot_clusters_dists_paper(title=title, xlim = xlim)

    def process_trial_regrets(self):
        policies_results = {}

        for policy_str, policy_params in self.driver.policy_inputs.items():
            policies_results[policy_str] = []
            for trial_num, trial in self.driver.trials.items():
                pol_results = trial["results"][policy_str]
                policies_results[policy_str].append(pol_results.sum_regret)
        pol_regrets = {}

        for policy_str, trial_results in policies_results.items():
            summary_df = pd.DataFrame()
            pol_regret_df = pd.concat([x["Expected"] for x in trial_results], axis = 1, keys = [f"Trial {i}" for i in range(len(trial_results))] )
            summary_df['mean'] = pol_regret_df.mean(axis=1)
            summary_df["median"] = pol_regret_df.median(axis=1)
            summary_df['std'] = pol_regret_df.std(axis=1)
            pol_regret_df = pd.concat([pol_regret_df, summary_df], axis = 1)
            pol_regrets[policy_str] = pol_regret_df
        avg_regret_df = pd.concat([x["mean"] for x in pol_regrets.values()], axis=1, keys = [i for i in pol_regrets.keys()])
        std_regret_df = pd.concat([x["std"] for x in pol_regrets.values()], axis=1, keys = [i for i in pol_regrets.keys()])
        self.pol_regrets = pol_regrets
        self.avg_regret_df = avg_regret_df
        self.std_regret_df = std_regret_df
        return pol_regrets, avg_regret_df, std_regret_df

    def plot_pol_avg_regrets(self, fig = None, ax = None, N = False, labels = None):
        #plt.rcParams['text.usetex'] = True
        if fig is None:
            fig, ax = plt.subplots()
        styles = ['-', '--']
        if N:
            for col, style in zip(self.avg_regret_df.columns, styles):
                if labels is None:
                    label = col + f" (N={self.driver.trials[0]['env'].nbClusters})"
                else:
                    label = labels.pop(0)
                self.avg_regret_df[col].plot(style = style, ax = ax, label = label)
        else:
            for col, style in zip(self.avg_regret_df.columns, styles):
                if labels is None:
                    label = col
                else:
                    label = labels.pop(0)
                self.avg_regret_df[col].plot(style = style, ax = ax, label = label)
        #ax.set_title("Average Regret")
        ax.set_xlabel("Round, $t$", size = 12)
        ax.set_ylabel("Regret, $\mathcal{R}(t,\Theta)$", size = 12)
        ax.legend(loc = 'upper left')
        return fig, ax

    def plot_pol_avg_regretsN(self, fig, ax):
        #plt.rcParams['text.usetex'] = True
        ax.plot(self.avg_regret_df, style = ['-', '.'], label = [x + f" - N={self.driver.trials[0]['env'].nbClusters}"
                                                                 for x in self.avg_regret_df.columns])
        ax.set_xlabel("Round, $t$", size = 12)
        ax.set_ylabel("Regret, $\mathcal{R}(t,\Theta)$", size = 12)
        ax.legend(loc = 'upper left')
        return fig, ax

    def summarize_results(self, print_all = False):
        pd.options.display.float_format = '{:.2f}'.format
        np.set_printoptions(precision=2)

        if self.pol_regrets is None:
            self.process_trial_regrets()
        for trial_num, trial in self.driver.trials.items():
            env = trial["env"]
            hyperprior_means = np.array(env.hyperprior_means)
            hyperprior_variances = np.array(env.hyperprior_variances)
            sample_cluster_means = np.array(env.cluster_means)
            sample_cluster_variances = np.array([x.sample_sigma for x in env.clusters])
            best_arm_means = np.array(env.best_means)
            variance_of_cluster_means = np.array(np.var(sample_cluster_means))
            average_cluster_variance = np.array(np.mean(sample_cluster_variances))
            arm_means = np.array([arm_mean for arm_means in env.arm_means for arm_mean in arm_means])
            total_mean_reward_variance = np.var(arm_means)
            if print_all:
                print('*'*80)
                print(f"Trial {trial_num}")
                print(f"Hyperprior Means: {hyperprior_means}")
                print(f"Hyperprior Variances: {hyperprior_variances}")
                print(f"Sample Cluster Means: {sample_cluster_means}")
                print(f"Sample Cluster Variances: {sample_cluster_variances}")
                print(f"Best Arm means per cluster: {best_arm_means}")
                print(f"Variance of cluster means: {variance_of_cluster_means:0.2f}")
                print(f"Average Cluster variance {average_cluster_variance:0.2f}")
                print(f"Overall Variance of arm rewards {total_mean_reward_variance:0.2f}")

        tails = {}
        for pol_name, pol_df in self.pol_regrets.items():
            tails[pol_name] = pol_df.tail(1)
        pol_performance = pd.concat(tails, axis = 0)
        pol_performance.droplevel(1)
        print(pol_performance)
        return pol_performance

    def plot_envs_n_regret(self):
        for trial_id, trial in self.driver.trials.items():
            fig, axes = plt.subplots(2,1)
            env = trial["env"]
            env.plot_clusters_dists(fig, axes[0], nrm = False, priors = True, gen = False, samp = True, best_arm=True)
            self.plot_trials_regret(fig, axes[1], trial_ids = [trial_id])

    def plot_trials_regret(self, fig = None, ax = None, trial_ids = None):
        if fig is None:
            fig, ax = plt.subplots()
        if trial_ids is None:
            trial_ids = list(self.driver.trials.keys())
        for trial_id in trial_ids:
            results = self.driver.trials[trial_id]['results']
            for result in results.values():
                result.plot_cumulative_regret(fig, ax)

    def plot_trial_decision_history(self, trial_id = 0):
        trial = self.driver.trials[trial_id]
        results = trial["results"]
        for pol_str, pol_results in results.items():
            pol_results.plot_decision_history()

    def plot_posterior_decision_history(self, trial_id = 0):
        fig, axes = plt.subplots(2,1)
        trial = self.driver.trials[trial_id]
        results = trial["results"]

    def compute_cluster_similarity(self, trial_id = 0):
        def KL_D(mu_1, mu_2, sig_1, sig_2):
            return np.sqrt(np.log(np.sqrt(sig_2/sig_1)) + (sig_1 + (mu_1-mu_2)**2)/(2*sig_2)-1/2)
        env = self.driver.trials[trial_id]["env"]
        KL = 0
        for i in range(env.nbClusters):
            for j in range(env.nbClusters):
                mu_1 = env.cluster_means[i]
                sig_1 = env.cluster_variances[i]
                mu_2 = env.cluster_means[j]
                sig_2 = env.cluster_variances[j]
                KL += -KL_D(mu_1, mu_2, sig_1, sig_2)
        return KL/(2*env.nbClusters)

    def compute_avg_cluster_similarity(self):
        KL = 0
        num = 0
        for trial_id, trial in self.driver.trials.items():
            KL += self.compute_cluster_similarity(trial_id)
            num +=1

        return KL/num

    def posteriorconvergence_plot(self, trial_id):
        fig, ax = plt.subplots()
        env = self.driver.trials[trial_id]["env"]
        HTS_results = self.driver.trials[trial_id]['results']['HTS']
        parameter_history = HTS_results.parameter_history
        for cluster_index, df in parameter_history.items():
            u_i = df['u_i']
            x = np.arange(0, len(u_i))
            v_i = df["v_i"]
            std_i = np.sqrt(v_i)
            CI = 2*std_i
            ax.plot(x, u_i, label=f"$u_{cluster_index}(t)$")
            upper_CI = u_i + CI
            lower_CI = u_i - CI
            ax.fill_between(x, lower_CI[0:], upper_CI[0:], alpha=0.2)
            ax.legend()
            ax.set_ylabel(f"Cluster Posterior Mean, $u_i(t)$")
            ax.set_xlabel(f"Round, $t$")

    def posteriorconvergence_plot2(self, trial_id, rounds = 50, everyn = None, startn = 0):
        fig, ax = plt.subplots()
        env = self.driver.trials[trial_id]["env"]
        if everyn is None:
            everyn = env.nbClusters+1

        HTS_results = self.driver.trials[trial_id]['results']['HTS']
        parameter_history = HTS_results.parameter_history
        dh_dfs = []
        dh = HTS_results.decision_history
        for cluster in range(env.nbClusters):
            dh_dfs.append(dh[dh['Cluster']==cluster])
        decision_rounds = [x.index.values for x in dh_dfs]
        for (cluster_index, df), dr in zip(parameter_history.items(), decision_rounds):
            dr = dr[dr<rounds]
            u_i = df['u_i'].values[0:rounds]
            x = np.arange(0, rounds)
            #inc = np.arange(0, len(u_i),5)
            v_i = df["v_i"].values[0:rounds]
            std_i = np.sqrt(v_i)
            CI = 2*std_i
            #err = CI[inc].values
            #ax.plot(x, u_i, label=f"$u_{cluster_index}(t)$")
            upper_CI = u_i + CI
            lower_CI = u_i - CI
            #ax.fill_between(x, lower_CI[0:], upper_CI[0:], alpha=0.2)
            ax.errorbar(x, u_i, yerr = CI, capsize = 3, elinewidth = 1, errorevery = (cluster_index + startn, everyn), label=f"$u_{cluster_index}(t)$")
            ax.legend()
            ax.set_ylabel(f"Cluster Posterior Mean, $u_i(t)$")
            ax.set_xlabel(f"Round, $t$")




def plot_merged_regrets(Helpers, N = False, labels = None, order  = None):
    fig, ax = plt.subplots()
    #dfs = [x['avg_regret_df'] for x in Helpers]
    for helper in Helpers:
        fig, ax = helper.plot_pol_avg_regrets(fig, ax, N = N, labels = labels)
    if order is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[idx]for idx in order], [labels[idx] for idx in order])
    else:
        ax.legend(loc = "center right")











Results_Folder = 'C:\\GitHub\\HMAB\\BHB\\NewPickledResults\\2022-07-20_15_44' # similarity scenarios
scenarios = [1,2,3] # for similarity scenarios


# Results_Folder = 'C:\\GitHub\\HMAB\\BHB\\NewPickledResults\\2022-07-20_21_23' # changing N scenarios
# scenarios = [1,2,3,4]



scenario = 1

SINGLE = False
MULTI = True
ENV_PLOT = False
if __name__ == "__main__":
    if ENV_PLOT:
        Load_Path = Results_Folder + f'\\scenario{scenario}.pkl'
        Driver = pickle.load(open(Load_Path, 'rb'))
        Helper = SimHelper(Driver)
        Helper.plot_environment_paper(0, xlim = None)
    if SINGLE:
        Load_Path = Results_Folder + f'\\scenario{scenario}.pkl'
        Driver = pickle.load(open(Load_Path, 'rb'))
        Helper = SimHelper(Driver)
        pol_regrets, avg_regrets_df, std_regrets_df = Helper.process_trial_regrets()
        # avg_regrets_df.plot()
        Helper.plot_envs([0])
        # Helper.plot_pol_avg_regrets()
        # pol_performance = Helper.summarize_results()
        # Helper.plot_envs_n_regret()
        #Helper.plot_trial_decision_history(trial_id=2)
        similarity = Helper.compute_cluster_similarity(0)
    if MULTI:
        Helpers = []
        Drivers = []
        avg_KL = []
        pol_performances = []
        for scenario in scenarios:
            Load_Path = Results_Folder + f'\\scenario{scenario}.pkl'
            Driver = pickle.load(open(Load_Path, 'rb'))
            Helper = SimHelper(Driver)
            Drivers.append(Driver)
            Helpers.append(Helper)
            pol_regrets, avg_regrets_df, std_regrets_df = Helper.process_trial_regrets()
            # avg_regrets_df.plot()
            #Helper.plot_envs([0])
            Helper.plot_pol_avg_regrets()
            pol_performances.append(Helper.summarize_results())
            # env_ids = [0]
            # for env_id in env_ids:
            #     Helper.plot_envs(env_id,title = f'Scenario {scenario} - env {env_id}')
            # Helper.plot_envs_n_regret()
            avg_KL.append(Helper.compute_avg_cluster_similarity())
            # Helper.plot_trial_decision_history(trial_id=2)
            print(avg_KL[-1])
            # Plot posterior convergence of trial x
            #Helper.posteriorconvergence_plot(0)
        # labels  = None #= ["HTS (Low)", "TS (Low)", "HTS (Mod)", "TS (Mod)", "HTS (High)", "TS (High)"]
        # order = [7,5,3,1, 6,4, 2,0]
        # plot_merged_regrets(Helpers, N = True, labels =labels, order = order)
        Helpers[1].posteriorconvergence_plot(0)
        Helpers[1].posteriorconvergence_plot2(0, rounds=1000, everyn=50, startn = 8)
