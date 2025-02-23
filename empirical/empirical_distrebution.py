import numpy as np


def ecdf(x, x_min=None, x_max=None):
    x = np.asarray(x)
    if len(x) == 0:
        values = np.array([])
        counts = np.array([])
    else:
        values, counts = np.unique(x, return_counts=True)
    
    if x_min is not None:
        if len(values) == 0 or x_min < values[0]:
            values = np.insert(values, 0, x_min)
            counts = np.insert(counts, 0, 0)
    
    if x_max is not None:
        if len(values) == 0 or x_max > values[-1]:
            values = np.append(values, x_max)
            counts = np.append(counts, 0)
    
    cum_counts = np.cumsum(counts)
    total = cum_counts[-1]
    e_freq = cum_counts / total
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
        idx_s = np.clip(idx_s, 0, n - 1)
        return e_freq[idx_s]
    return cdf


def get_rv_cdf_sorted(n):
    rv_cdf = np.random.rand(n)
    rv_cdf = np.sort(rv_cdf)
    k = 0
    for i in range(n):
        if rv_cdf[i] == 0:
            k = k + 1
        else:
            break
    if k > 0:
        rv_cdf[0:k] = get_rv_cdf_sorted(k)
        rv_cdf = np.sort(rv_cdf)
    return rv_cdf


def ecdf_rvs(values, e_freq, n):
    rv_cdf = get_rv_cdf_sorted(n)
    rv = np.zeros(np.shape(rv_cdf))
    
    k = 0
    for i in range(n):
        while rv_cdf[i] > e_freq[k]:
            k = k + 1
        rv[i] = values[k]
    return rv


def lcdf_rvs(values, e_freq, n):
    rv_cdf = np.random.rand(n)
    rv_cdf = np.sort(rv_cdf)
    rv = np.zeros(np.shape(rv_cdf))
    k = 0
    for i in range(n):
        while rv_cdf[i] > e_freq[k]:
            k = k + 1
        rv[i] = (values[k] - values[k - 1]) / (e_freq[k] - e_freq[k - 1]) * (rv_cdf[i] - e_freq[k - 1]) + values[k - 1]
    return rv
