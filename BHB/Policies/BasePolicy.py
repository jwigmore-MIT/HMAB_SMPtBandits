from scipy.stats import norm
import numpy as np
from BHB.config import DEBUGSETTINGS as DS
from copy import deepcopy as dc

class BasePolicy(object):

    def __init__(self, estimator, name = None, type = None):
        self.estimator = dc(estimator)
        self.type = type
        self.name = name
        self.nbClusters = estimator.nbClusters
        self.nbArms = estimator.nbArms
        self.id = None
        self.rng = estimator.rng

    def __str__(self):
        return f"Policy: " + self.name

    def get_1D_arm_index(self, cluster_index, arm_index_2D):
        return np.ravel_multi_index((cluster_index, arm_index_2D), (self.nbArms, self.nbClusters))

    def get_2D_arm_index(self, arm_index_1D):
        return np.unravel_index(arm_index_1D, (self.nbClusters, self.nbArms))


