from BHB.Policies.BasePolicy import BasePolicy
from scipy.stats import norm
import numpy as np
from BHB.config import DEBUGSETTINGS as DS
from copy import deepcopy as dc

class UCB(BasePolicy):
    '''
    Classical UCB Implementation
    '''
    def __init__(self, estimator, name):
        super().__init__(estimator,  name, type = "UCB")


    def choice(self, t = None):
        self.estimator.compute_indices(t)
        max_bb_index, max_arm_index, max_ucb = self.estimator.getMaxIndex()
        return max_bb_index

