import numpy as np


def ecdf(x, x_min=None, x_max=None):
    if x_min is not None:
        x = np.append(x, x_min)
    values, counts = np.unique(x, return_counts=True)
    counts[0] = 0
    cum_counts = np.cumsum(counts)
    e_freq = cum_counts / cum_counts[-1]
    return values, e_freq


def empirical_cdf_gen(values, e_freq, return_linear=False):
    n = len(values)
    if n != len(e_freq):
        raise ValueError("Length of 'values' and 'e_freq' must be the same.")
    
    args = np.argsort(values)
    values = values[args]
    e_freq = e_freq[args]
    
    if return_linear is True:
        slopes = np.zeros(n)
        slopes[0:-1] = np.diff(e_freq) / np.diff(values)
        
        def cdf(x):
            idx_s = np.searchsorted(values, x, "left")
            idx_s = np.clip(idx_s, 0, n - 1)
            y = slopes[idx_s]*(x - values[idx_s]) + e_freq[idx_s]
            return y
        return cdf
    
    def cdf(x):
        idx_s = np.searchsorted(values, x, "left")
        return e_freq[idx_s]
    return cdf


def lcdf_rvs(x, cdf, n):
    rv_cdf = np.random.rand(n)
    rv_cdf = np.sort(rv_cdf)
    rv = np.zeros(np.shape(rv_cdf))
    k = 1
    for i in range(0, n):
        while rv_cdf[i] > cdf[k]:
            k = k + 1
        rv[i] = (x[k] - x[k - 1]) / (cdf[k] - cdf[k - 1]) * (rv_cdf[i] - cdf[k - 1]) + x[k - 1]
    return rv
