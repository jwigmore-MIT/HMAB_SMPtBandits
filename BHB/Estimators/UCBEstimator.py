from typing import Union
from BHB.helper import *

import numpy as np
from BHB.Estimators.BaseEstimator import BaseEstimator
from BHB.config import DEBUGSETTINGS as DS

np.seterr(all="raise")


class UCBEstimator(BaseEstimator):

    def __init__(self, environment, Settings, type_overide = None):
        # If delta is None, use Algorithm 6 in Clusterit Algorithms (Lattimore,2020)
        self.environment = environment
        self.nbClusters = environment.nbClusters
        self.nbArms = environment.nbArms
        self.Type = Settings["UCB_Type"]
        if self.Type == "delta":
            self.delta = Settings["UCB_delta"]
        if type_overide is not None:
            self.Type = type_overide
        self.UCB_indices = np.ones([self.nbArms * self.nbClusters]) * np.infty
        self.mu_hat = np.ones([self.nbArms * self.nbClusters])
        self.counts = np.zeros([self.nbArms * self.nbClusters])
        self.sums = np.zeros([self.nbArms * self.nbClusters])
        self.rng = environment.rng

    def compute_indices(self, time):
        for arm_num in range(self.nbArms*self.nbClusters):
            if self.counts[arm_num] == 0:
                return
            if self.Type == "anytime":
                '''Asymptotically Optimal UCB - See Lattimore pg 116'''
                self.UCB_indices[arm_num] = self.mu_hat[arm_num] + np.sqrt(
                    2 * np.log(1 + time * np.log(time) ** 2) / self.counts[arm_num])
            elif self.Type == "delta":
                '''Classic UCB based on confidence level parameter $\delta$
                   See Lattimore pg. 102'''
                self.UCB_indices[arm_num] = self.mu_hat[arm_num] + np.sqrt(
                    2 * np.log(1 / self.delta) / self.counts[arm_num])
            elif self.Type == "MOSS":
                ''' Minimax optimal UCB - See Lattimore pg. 122'''
                self.UCB_indices[arm_num] = self.mu_hat[arm_num] + np.sqrt(4 / (self.counts[arm_num])) \
                                        * np.log(
                    np.max([self.environment.horizon / (self.nbArms * self.nbClusters * self.counts[arm_num]), 1]))


    def update(self, time, cluster_index, arm_index, reward):
        if arm_index is not None:
            arm_index = self.get_1D_arm_index(cluster_index, arm_index)
        if time < self.nbClusters:
            return
        self.counts[arm_index] += 1
        self.sums[arm_index] += reward
        self.mu_hat[arm_index] = self.sums[arm_index] / self.counts[arm_index]


    def getMaxIndex(self):
        max_index = np.argmax(self.UCB_indices)
        max_value = self.UCB_indices[max_index]
        max_bb_index = self.get_2D_arm_index(max_index)
        return (max_bb_index, max_index, max_value)

    def get_latest_params(self):
        return "NOT IMPLEMENTED"
