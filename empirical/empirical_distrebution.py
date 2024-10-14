import numpy as np


def ecdf(x, x_min=None, x_max=None):
	if x_min is not None:
		x = np.append(x, x_min)
	x, c = np.unique(x, return_counts=True)
	c[0] = 0
	cdf = np.cumsum(c)
	cdf = cdf / cdf[-1]
	return x, cdf


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
