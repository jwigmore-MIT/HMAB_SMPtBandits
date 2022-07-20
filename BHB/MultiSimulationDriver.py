'''
Want to run multiple simulations back to back under different environments
Inputs: Environment Settings, Policies
'''
import numpy as np
from BHB.Estimators.KnownVarianceEstimator import KnownVarianceEstimator
from BHB.Estimators.ThompsonSamplingEstimator import  ThompsonSamplingEstimator
from BHB.Estimators.UCBEstimator import UCBEstimator
from BHB.Estimators.BayesUCBEstimator import BayesUCBEstimator
from BHB.PG_Environment import PG_Environment
from copy import deepcopy
from BHB.Test_Scripts.HTS_TS_Scenarios2_7_19 import scenario1, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7
from BHB.Evaluator import *
from BHB.config import DEBUGSETTINGS as DS
import pickle
from datetime import datetime
import os

import addcopyfighandler





from BHB.Policies.HierarchicalThompsonSampling import Full_HTS, Tiered_HTS, Full_Sample_HTS
from BHB.Policies.UCB import UCB
from BHB.Policies.ThompsonSampling import TS


class MultiSimulationDriver(object):

    def __init__(self, multi_settings):
        self.nbTrials = multi_settings["nbTrials"]
        self.policy_inputs = multi_settings["policies"]
        self.env_settings = multi_settings["env_settings"]
        self.rng = multi_settings['rng']
        self.trials = {}
        self.init_trials()


    def init_trials(self):
        self.env_seeds = self.rng.integers(100000, size = self.nbTrials)
        for n in range(self.nbTrials):
            self.trials[n] = {}
            env_settings  = deepcopy(self.env_settings)
            env_settings["rng"] = np.random.default_rng(self.env_seeds[n])
            env = PG_Environment(env_settings, n)
            self.trials[n]['env'] = env
            self.trials[n]["results"] = {}
            self.init_policies(env, n)



    def init_policies(self, env, n):
        self.trials[n]["policies"] = {}
        for pol_str, pol_params in self.policy_inputs.items():
            if pol_str == "HTS":
                est = KnownVarianceEstimator(env, pol_params["est_priors"])
                pol = Full_HTS(est, name = f"{pol_str} - Env {n}")
                self.trials[n]["policies"][pol_str] = pol
                self.trials[n]["results"][pol_str] = None
            if pol_str == "Full HTS UI":
                est = KnownVarianceEstimator(env, "Uninformative")
                pol = Full_HTS(est, name = f"Full HTS UI - Env {n}")
                self.trials[n]["policies"]["Full HTS UI"] = pol
                self.trials[n]["results"]["Full HTS UI"] = None
            if pol_str == "Full Sample HTS":
                est = KnownVarianceEstimator(env, pol_params["est_priors"])
                pol = Full_Sample_HTS(est, name = f"Full Sample HTS - Env {n}")
                self.trials[n]["policies"]["Full Sample HTS"] = pol
                self.trials[n]["results"]["Full Sample HTS"] = None
            if pol_str == "Tiered HTS":
                est = KnownVarianceEstimator(env, pol_params["est_priors"])
                pol = Tiered_HTS(est, name=f"Tiered HTS - Env {n}")
                self.trials[n]["policies"]["Tiered HTS"] = pol
                self.trials[n]["results"]["Tiered HTS"] = None
            if pol_str == "Tiered HTS UI":
                est = KnownVarianceEstimator(env, "Uninformative")
                pol = Tiered_HTS(est, name=f"Tiered HTS UI - Env {n}")
                self.trials[n]["policies"]["Tiered HTS UI"] = pol
                self.trials[n]["results"]["Tiered HTS UI"] = None
            if pol_str == "Bayes UCB":
                est = BayesUCBEstimator(env, self.env_settings, est_priors= pol_params["est_priors"])
                pol = UCB(est, name = "Bayes UCB")
                self.trials[n]["policies"]["Bayes UCB"] = pol
                self.trials[n]["results"]["Bayes UCB"] = None
            if pol_str == "MOSS UCB":
                est = UCBEstimator(env, pol_params)
                pol = UCB(est, name='UCB - MOSS')
                self.trials[n]["policies"]["MOSS UCB"] = pol
                self.trials[n]["results"]["MOSS UCB"] = None
            if pol_str == "Delta UCB":
                est = UCBEstimator(est, pol_params)
                pol = UCB(est, name='UCB - Delta')
                self.trials[n]["policies"]["Delta UCB"] = pol
                self.trials[n]["results"]["Delta UCB"] = None
            if pol_str == "TS":
                est = ThompsonSamplingEstimator(env, pol_params["est_priors"])
                pol = TS(est, name = f"TS - Env{n}")
                self.trials[n]["policies"]["TS"] = pol
                self.trials[n]["results"]["TS"] = None
            if pol_str == "TS UI":
                est = ThompsonSamplingEstimator(env, "Uninformative")
                pol = TS(est, name = f"TS - Env{n}")
                self.trials[n]["policies"]["TS UI"] = pol
                self.trials[n]["results"]["TS UI"] = None





    def run_trials(self, trial_ids = None, summary = False):
        trials_results = []
        if trial_ids is None:
            trial_ids = list(self.trials.keys())
        for trial_id in trial_ids:
            trials_results.append(self.run_trial(trial_id, summary))
        return trials_results

    def run_trial(self, trial_id = 0, summary = False):
        trial = self.trials[trial_id]
        pols = trial['policies']
        subtrial_results = []
        for pol_name in pols.keys():
            subtrial_results.append(self.run_subtrial(trial_id, pol_name, summary))
        return subtrial_results


    def run_subtrial(self, trial_id = '', pol_id = '', summary = False,  results_folder = None):
        '''
        Runs single policy on a single environment
        :param trial_id:
        :param pol_id:
        :param results_folder:
        :return:
        '''
        start=timer()
        environment = self.trials[trial_id]["env"]
        horizon = environment.horizon
        policy = self.trials[trial_id]["policies"][pol_id]
        results_type = policy.type
        estimator = policy.estimator
        t = 0
        r_name = f"Env/Pol = [{trial_id}/{pol_id}]"
        if results_type == "BHB":
            results = HMAB_Results(environment, policy, horizon, r_name)
        if results_type == "UCB":
            results = UCB_Results(environment, policy, horizon, r_name)
        while t < horizon:

            (band, bin) = policy.choice(t)

            observation = environment.draw_samples(band, bin, 1)
            estimator.update(t, band, bin, observation)
            if DS["play_game"]:

                print(f"Chosen band/bin = {band}/{bin}")
                print(f"Observed Reward = {observation[0]:.2f}")
            # store choice and observation in result

            # pass choice and observation to the estimator
            latest_params = estimator.get_latest_params()
            results.store(band, bin, observation, latest_params)


            t = t+1
            if DS["play_game"]: print("*"*20)
        end = timer()
        results.run_time = end-start
        results.finalize_results()
        if summary:
            results.summarize_results()
        if results_folder is not None:
            results.store_results(results_folder)
        self.trials[trial_id]["results"][pol_id] = results
        return results



class SimHelper(object):

    def __init__(self, driver):
        self.driver = driver
        self.pol_regrets = None
        self.avg_regret_df = None

    def plot_all_envs(self):
        for trial in self.driver.trials.values():
            env = trial["env"]
            env.plot_clusters_dists(nrm = False, priors = True, gen = False, samp = True)

    def plot_envs(self, env_ids):
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        for env_id in env_ids:
            env = self.driver.trials[env_id]["env"]
            env.plot_clusters_dists(nrm=False, priors=True, gen=False, samp=True)

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

    def plot_pol_avg_regrets(self):
        #plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots()
        ax.plot(self.avg_regret_df, label = [x for x in self.avg_regret_df.columns])
        #ax.set_title("Average Regret")
        ax.set_xlabel("Round, $t$", size = 12)
        ax.set_ylabel("Regret, $\mathcal{R}(t,\Theta)$", size = 12)
        ax.legend(loc = 'upper left')

    def summarize_results(self):
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







scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7]
scenario = scenario1


RUN_LOAD = False# Run and load a single scenario
RUN_MULTI = True # Run Multiple scenarios
RUN = False # Run a single scenario
LOAD = False # Load a single scenario


# Load_Path = 'C:\GitHub\HMAB\BHB\Pickled_Trials\\2022-07-19_10_55.pkl' # Scenario 3, Strong overlap
# Load_Path = 'C:\\GitHub\\HMAB\\BHB\\Pickled_Trials\\2022-07-19_17_34.pkl' # Scenar 2, Moderate overlap
Load_Path = 'C:\GitHub\HMAB\BHB\Pickled_Trials\scenario6_2022-07-19_23_35.pkl' #

if __name__ == "__main__":
    if RUN:
        time = datetime.now().strftime("%Y-%m-%d_%H_%M")
        Store_Path = f'C:\\Users\\Jerrod-CNRG\\Documents\\GitHub\\HMAB_SMPtBandits\\BHB\\NewPickledResults\\{time}.pkl'
        settings = scenario
        Driver = MultiSimulationDriver(settings)
        all_results = Driver.run_trials(summary = True)
        pickle.dump(Driver, open(Store_Path, 'wb'))
    if LOAD:
        Driver = pickle.load(open(Load_Path, 'rb'))
        Helper = SimHelper(Driver)
        pol_regrets, avg_regrets_df, std_regrets_df = Helper.process_trial_regrets()
        # avg_regrets_df.plot()
        Helper.plot_envs([0])
        Helper.plot_pol_avg_regrets()
        pol_performance = Helper.summarize_results()
        Helper.plot_envs_n_regret()
        #Helper.plot_trial_decision_history(trial_id=2)
    if RUN_LOAD:
        time = datetime.now().strftime("%Y-%m-%d_%H_%M")
        Store_Path = f'C:\\Users\\Jerrod-CNRG\\Documents\\GitHub\\HMAB_SMPtBandits\\BHB\\NewPickledResults\\{time}.pkl'
        settings = scenario
        Driver = MultiSimulationDriver(settings)
        all_results = Driver.run_trials(summary=True)
        pickle.dump(Driver, open(Store_Path, 'wb'))
        Driver = pickle.load(open(Store_Path, 'rb'))
        Helper = SimHelper(Driver)
        pol_regrets, avg_regrets_df, std_regrets_df = Helper.process_trial_regrets()
        # avg_regrets_df.plot()
        Helper.plot_pol_avg_regrets()
        pol_performance = Helper.summarize_results()
        Helper.plot_envs_n_regret()
    if RUN_MULTI:
        i = 1
        time = datetime.now().strftime("%Y-%m-%d_%H_%M")
        os.mkdir(f'C:\\Users\\Jerrod-CNRG\\Documents\\GitHub\\HMAB_SMPtBandits\\BHB\\NewPickledResults\\{time}')
        for scenario in scenarios:
            Store_Path = f'C:\\Users\\Jerrod-CNRG\\Documents\\GitHub\\HMAB_SMPtBandits\\BHB\\NewPickledResults\\{time}\\scenario{i}.pkl'
            settings = scenario
            Driver = MultiSimulationDriver(settings)
            all_results = Driver.run_trials(summary=True)
            pickle.dump(Driver, open(Store_Path, 'wb'))
            i += 1

