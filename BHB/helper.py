import numpy as np

def get_1D_arm_index(self, cluster_index, arm_index_2D):
    return np.ravel_multi_index((cluster_index, arm_index_2D), (self.nbClusters, self.nbArms))


def get_2D_arm_index(self, arm_index_1D):
    return np.unravel_index(arm_index_1D, (self.nbClusters, self.nbArms))