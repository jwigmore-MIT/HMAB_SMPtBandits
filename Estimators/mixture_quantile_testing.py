import numpy as np
from scipy.stats import norm




# Return the smallest value x between lo and hi such that f(x) >= v
def continuous_bisect_fun_left(f, v, lo, hi):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for i in range(32):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k

# Return the function that is the cdf of the mixture distribution
def get_mixture_cdf(component_distributions, ps):
    return lambda x: sum(component_dist.cdf(x) * p for component_dist, p in zip(component_distributions, ps))

# Return the pth quantile of the mixture distribution given by the component distributions and their probabilities

def mixture_quantile(p, component_distributions, ps):
    mixture_cdf = get_mixture_cdf(component_distributions, ps)

    # We can probably be a bit smarter about how we pick the limits
    lo = np.min([dist.ppf(p) for dist in component_distributions])
    hi = np.max([dist.ppf(p) for dist in component_distributions])

    return continuous_bisect_fun_left(mixture_cdf, p, lo, hi)


component_dists = [norm(0,1), norm(0.25, 1)]
mix = [0.5, 0.5]
p = 0.75
quantile = mixture_quantile(p,component_dists,mix)

