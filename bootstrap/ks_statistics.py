import numpy as np
from pyrockstats import ecdf
from pyrockstats.empirical import lecdf_rvs
from pyrockstats.empirical.empirical_distrebution import empirical_cdf_gen
from matplotlib import pyplot as plt


def ks_norm(f1, f2):
	assert len(f1) == len(f2), "lengths of the functions must be equal"
	return np.max(np.abs(f1 - f2))


def get_pseudo_ks(x, model, xmin=None, xmax=None):
	theta = model.fit(x, xmin=xmin, xmax=xmax)
	dist = model(*theta)
	rv = dist.rvs(len(x), xmin=xmin, xmax=xmax)
	pseudo_values, pseudo_freqs = ecdf(rv, xmin=xmin, xmax=xmax)
	boot_cdf = dist.cdf(pseudo_values, xmin=xmin, xmax=xmax)
	# plt.plot(pseudo_values, boot_cdf)
	# plt.plot(pseudo_values, pseudo_freqs)
	# plt.xscale('log')
	# plt.show()
	# exit()
	return ks_norm(boot_cdf, pseudo_freqs)


def get_ks_distribution(x: np.ndarray, model, n_ks=100, xmin=None, xmax=None):
	xmin = np.min(x) if xmin is None else xmin
	xmax = np.max(x) if xmax is None else xmax
	n = len(x)
	values, freqs = ecdf(x)
	ks = np.empty(n_ks)
	for i in range(n_ks):
		boot_x = lecdf_rvs(values, freqs, n)
		ks[i] = get_pseudo_ks(boot_x, model, xmin=xmin, xmax=xmax)
	return ks
