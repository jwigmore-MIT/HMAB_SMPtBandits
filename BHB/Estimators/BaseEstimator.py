
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

class BaseEstimator(object):

    def __init__(self, environment):
        self.env = environment
        self.nbClusters = environment.nbClusters
        self.nbArms = environment.nbArms

    def get_1D_arm_index(self, cluster_index, arm_index_2D):
        return np.ravel_multi_index((cluster_index, arm_index_2D), (self.nbClusters, self.nbArms))

    def get_2D_arm_index(self, arm_index_1D):
        return np.unravel_index(arm_index_1D, (self.nbClusters, self.nbArms))
