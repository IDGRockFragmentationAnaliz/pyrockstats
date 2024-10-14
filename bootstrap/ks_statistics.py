import numpy as np
from pyrockstats import ecdf
from pyrockstats.empirical import lcdf_rvs


def get_ks_distribution(s: np.ndarray, model, n_ks=100, x_min=None, x_max=None):
	x_min = np.min(s) if x_min is None else x_min
	x_max = np.max(s) if x_max is None else x_max
	n = len(s)
	x, cdf = ecdf(s)
	ks = np.empty(n_ks)
	for i in range(n_ks):
		_s = lcdf_rvs(x, cdf, n)
		_x, _ecdf = ecdf(s, x_min=x_min)
		theta = model.fit(_s)
		distribution = model(*theta)
		ks[i] = np.max(np.abs(_ecdf - distribution.cdf(_x)))
	return ks
