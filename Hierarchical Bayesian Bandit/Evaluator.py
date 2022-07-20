from Result import *
from config import DEBUGSETTINGS as DS
from copy import deepcopy as dc



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


    def dom_check(self,t, estimator):
        best_band, is_dom, a_i = estimator.test_posterior_dominance(delta=0.05, gran=1000)
        if DS["dom_check"]:
            print("*" * 20)
            print(f"Round {t}")
            post_strs = estimator.get_band_posterior_strs(printit = True)
            print(f"Best Band: {best_band}")
            print(f"Is Dom?: {is_dom}")
            print(f"a_i : {a_i}")
        return best_band, is_dom, a_i

    def play_game(self, horizon, env_id = 0, pol_id = 0, results_type = "HMAB" , results_folder = None):
        environment = self.environments[env_id]
        policy = self.policies[pol_id]
        estimator = policy.estimator
        t = 0
        r_name = f"Env/Pol = [{env_id}/{pol_id}]"
        isnt_dom = True
        if results_type == "HMAB":
            results = HMAB_Results(environment, policy, horizon, r_name)
        if results_type == "UCB":
            results = UCB_Results(environment, policy, horizon, r_name)
        while t < horizon:
            (band, bin, k) = policy.choice(t)

            observation = environment.draw_samples(band, bin, k)
            estimator.update(t, band, bin, observation)
            if DS["play_game"]:

                print(f"Chosen band/bin = {band}/{bin}")
                print(f"Observed Reward = {observation[0]:.2f}")
            # store choice and observation in result

            # pass choice and observation to the estimator
            latest_params = estimator.get_latest_params()
            if results_type == "HMAB":
                if isnt_dom:
                    dom_results = estimator.test_posterior_dominance(delta=.01, gran=1000)
                    results.store_dom_results(dom_results)
                    if dom_results[1] > 0:
                        isnt_dom = False
            results.store(band, bin, observation, latest_params)


            t = t+k
            if DS["play_game"]: print("*"*20)

        results.finalize_results()
        if results_folder is not None:
            results.store_results(results_folder)
        return results






