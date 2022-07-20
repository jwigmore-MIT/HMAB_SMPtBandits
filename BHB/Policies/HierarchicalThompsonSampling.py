from BHB.Policies.BasePolicy import BasePolicy
from scipy.stats import norm
import numpy as np
from BHB.config import DEBUGSETTINGS as DS
from copy import deepcopy as dc
from BHB.helper import *




class Full_HTS(BasePolicy):
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

class Full_Sample_HTS(BasePolicy):
    '''
    Full Hierarchical Thompson sampling where the posteriors of all arms are sampled from on everytime step
    '''
    def __init__(self, estimator, name = None):
        super().__init__(estimator,  name, type = "BHB")
        self.estimator = estimator



    def choice(self, t= None):
        if t < self.nbClusters*self.nbArms:
            clustarm_index = self.get_2D_arm_index(t)
            return clustarm_index
        sampled_arm_params = self.estimator.full_hier_sampling()
        best_arm_1D_index = sampled_arm_params.argmax()
        return self.get_2D_arm_index(best_arm_1D_index)




class Tiered_HTS(BasePolicy):

    def __init__(self, estimator, name = None):
        super().__init__(estimator,  name, type = "BHB")
        self.estimator = estimator


    def choice(self, t = None):
        chosen_cluster_index, theta_i_star = self.choose_cluster()
        chosen_arm_index = self.choose_arm(chosen_cluster_index, theta_i_star)
        return (chosen_cluster_index, chosen_arm_index)


    def choose_cluster(self):
        # Sample from cluster beliefs
        theta_i_samples = self.estimator.sample_cluster_instances()
        chosen_cluster_index = theta_i_samples.argmax()
        theta_i_star = theta_i_samples.max()
        return chosen_cluster_index, theta_i_star

    def choose_arm(self, cluster_index, theta_i_star):
        theta_ij_samples = self.estimator.sample_arm_instances(cluster_index, theta_i_star)
        chosen_arm_index = theta_ij_samples.argmax()
        return chosen_arm_index


