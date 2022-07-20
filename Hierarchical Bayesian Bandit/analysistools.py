import numpy as np
from scipy.stats import norm
import pandas as pd


def compute_a_i(f,F):
    '''
    Computes a_i by first taking the log of the inputs
    :param f:
    :param F:
    :param f_bar:
    :return:
    '''
    l1 = np.log(f)
    l2 = np.log(F)
    l3 = np.sum(F, axis = 1, keepdims= True)
    l = l1+l3-l2
    e = np.exp(l)
    s = np.sum(e, axis = 0)
    tru_ai = s/np.sum(s)
    return tru_ai



def pr_max_X(X, M = 1000, p = 0.01):
    '''Computes the probability X_i (r.v.) in X = [X_1, ..., X_N] is the
       max of the set using M samples from all X_i
       p is the lower tail probability ... (1-p) would be upper tail
       Has problems when there are large differences in the support of X_i's -- UNDERFLOW FLOATING POINT ERRORS
    '''
    N = len(X)
    # Need M points over the support of all N distributions
    min_x = np.infty # min of support
    max_x = -np.infty # max of support
    for X_i in X:
        min_x = min(X_i.ppf(p), min_x) # check if min of support is less than current min_x
        max_x = max(X_i.ppf(1-p), max_x)
    x = np.linspace(min_x, max_x, num = M)


    f = np.zeros([M, N]) # pdf evaluated at M different points for each X_i (col)
    F = np.zeros([M, N]) # cdf evaluated at M different points for each X_i (col)
    for i in range(N):
        f[:,i] = X[i].pdf(x)
        F[:,i] = X[i].cdf(x)
    tru_ai = compute_a_i(f,F)
    # F_bar = np.prod(F, axis = 1, keepdims= True)
    # try:
    #     temp = np.multiply(np.divide(F_bar,F), f)
    # except RuntimeWarning:
    #     print(RuntimeWarning)
    # a_i = np.sum(temp, axis = 0) # unnormalized probability of being the max
    # tru_ai = a_i/np.sum(a_i) # Normalized
    return tru_ai

def sim_pr_max(X, M = 100_000):
    draws = np.zeros([M, len(X)])
    i = 0
    for X_i in X:
        draws[:, i] = X_i.rvs(size=M).T
        i += 1
    is_max = pd.DataFrame(np.argmax(draws, axis=1))
    a_i = []
    for j in range(i):
        a_i.append((is_max.values == j).sum()/M)
    return a_i




if __name__ == '__main__':
    X1 = norm(2.8,1)
    X2 = norm(3,1)
    X3 = norm(3.2,1)
    X4 = norm(3.4,1)
    X = [X1, X2, X3, X4]

    tru_ai = pr_max_X(X, 100_000)

    a_i_sim = sim_pr_max(X, 5_000_000)



