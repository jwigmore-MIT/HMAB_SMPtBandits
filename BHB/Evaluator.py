from BHB.Result import *
from BHB.config import DEBUGSETTINGS as DS
from BHB.helper import *
from copy import deepcopy as dc
from timeit import default_timer as timer



class Evaluator(object):

    def __init__(self, environments, policies):

        if hasattr(environments, '__iter__'):
            self.environments = [dc(x) for x in environments]
        else:
            self.environments = [environments]
        if hasattr(policies, '__iter__'):
            self.policies = [dc(x) for x in policies]
        else:
            self.policies = [dc(policies)]

        self.all_results = []


    def play_game(self, env_id = 0, pol_id = 0, results_folder = None):
        start=timer()
        environment = self.environments[env_id]
        horizon = environment.horizon
        policy = self.policies[pol_id]
        results_type = policy.type
        estimator = policy.estimator
        t = 0
        r_name = f"Env/Pol = [{env_id}/{pol_id}]"
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
        if results_folder is not None:
            results.store_results(results_folder)
        self.all_results.append(results)
        return results

    def play_games(self, env_ids = None, pol_ids = None, results_folder = None):
        if env_ids is None:
            env_ids = [i for i in range(len(self.environments))]
        if pol_ids is None:
            pol_ids = [i for i in range(len(self.policies))]

        for env_id in env_ids:
            for pol_id in pol_ids:
                self.play_game(env_id, pol_id, results_folder)

    def plot_decision_histories(self):
        for result in self.all_results:
            result.plot_decision_history()

    def print_summaries(self, print_arms = True):
        for result in self.all_results:
            result.summarize_results(arms = print_arms)

    def plot_cumulative_regrets(self):
        fig, ax = plt.subplots()
        for result in self.all_results:
            result.plot_cumulative_regret(ax = ax)









