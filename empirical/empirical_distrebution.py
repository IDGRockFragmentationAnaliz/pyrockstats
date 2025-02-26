import numpy as np


def ecdf(x, xmin=None, xmax=None):
    x = np.asarray(x)
    if len(x) == 0:
        values = np.array([])
        counts = np.array([])
    else:
        values, counts = np.unique(x, return_counts=True)
    
    if xmin is not None:
        if len(values) == 0 or xmin < values[0]:
            values = np.insert(values, 0, xmin)
            counts = np.insert(counts, 0, 0)
    
    if xmax is not None:
        if len(values) == 0 or xmax > values[-1]:
            values = np.append(values, xmax)
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
    e_freq = (e_freq - e_freq[0]) / (e_freq[-1] - e_freq[0])
    idx_s = np.searchsorted(e_freq, rv_cdf, "left") - 1
    rv = values[idx_s]
    return rv


def lecdf_rvs(values, e_freq, n):
    rv_cdf = get_rv_cdf_sorted(n)
    rv_sub_cdf = np.random.rand(n)
    d_values = np.diff(values)
    e_freq = (e_freq - e_freq[0]) / (e_freq[-1] - e_freq[0])
    idx_s = np.searchsorted(e_freq, rv_cdf, "left")-1
    rv = d_values[idx_s]*rv_sub_cdf + values[idx_s]
    return rv
