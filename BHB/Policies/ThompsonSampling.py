from BHB.Policies.BasePolicy import BasePolicy
from scipy.stats import norm
import numpy as np
from BHB.config import DEBUGSETTINGS as DS
from copy import deepcopy as dc
from BHB.helper import *




class TS(BasePolicy):
    '''
    Full Hierarchical Thompson sampling where the posteriors of all arms are sampled from on everytime step
    '''
    def __init__(self, estimator, name = None):
        super().__init__(estimator,  name, type = "BHB")
        self.estimator = estimator


    def choice(self, t= None):
        sampled_arm_params = self.estimator.full_hier_sampling()
        best_arm_1D_index = sampled_arm_params.argmax()
        return self.get_2D_arm_index(best_arm_1D_index)



