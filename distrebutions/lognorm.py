import math
import numpy as np
from scipy.special import erf
from scipy.special import gammainc
from scipy.special import gamma
from scipy.optimize import minimize
from .mle import MLEModification


def pdf(x, sigma, scale):
	return (1 / (x * sigma * np.sqrt(2 * math.pi))) * np.exp(-0.5 * ((np.log(x / scale)) / sigma) ** 2)


def cdf(x, sigma, scale):
	return 0.5 * (1 + erf(np.log(x / scale) / (sigma * math.sqrt(2))))


def mean_logpdf(x, sigma, scale, mean_lnx=None, mean_lnx2=None):
	mean_lnx = mean_lnx if mean_lnx is not None else np.mean(np.log(x))
	mean_lnx2 = mean_lnx2 if mean_lnx2 is not None else np.mean(np.log(x) ** 2)
	mu = math.log(scale)
	part1 = -np.log(sigma) - mean_lnx2 / (2 * (sigma ** 2))
	part2 = (2 * mu * mean_lnx - mu ** 2) / (2 * (sigma ** 2))
	return part1 + part2


def get_functional(x, xmin=None, xmax=None, mean_lnx=None, mean_lnx2=None):
	modification = MLEModification(cdf, xmin=xmin, xmax=xmax)

	def functional(theta):
		l1 = -mean_logpdf(x, *theta, mean_lnx=mean_lnx, mean_lnx2=mean_lnx2)
		l2 = modification(theta)
		return l1 + l2

	return functional


def fit_mle(x, xmin=None, xmax=None):
	mx = np.mean(x)
	x = x / mx
	xmin = xmin / mx if xmin is not None else xmin
	xmax = xmax / mx if xmax is not None else xmax

	mean_lnx = np.mean(np.log(x))
	mean_lnx2 = np.mean(np.log(x) ** 2)

	theta_0 = np.array([1, 1])
	res = minimize(
		get_functional(x, xmin=xmin, xmax=xmax, mean_lnx=mean_lnx, mean_lnx2=mean_lnx2),
		theta_0,
		bounds=((1e-3, None), (1e-3, None)),
		method='nelder-mead',
		tol=1e-3
	)

	return res.x[0], res.x[1]*mx


class lognorm:
	def __init__(self, sigma, scale):
		self.sigma = sigma
		self.scale = scale
		self.mu = math.log(scale)

	def pdf(self, x):
		return pdf(x, self.sigma, self.scale)

	def cdf(self, x, xmin=None, xmax=None):
		if xmin is None and xmax is None:
			return cdf(x, self.sigma, self.scale)
		cdf_min = cdf(xmin, self.sigma, self.scale) \
			if xmin is not None else 0
		cdf_max = cdf(xmax, self.sigma, self.scale)\
			if xmax is not None else 1
		
		return (cdf(x, self.sigma, self.scale) - cdf_min)/(cdf_max - cdf_min)


	def mean_logpdf(self, x, mean_lnx2):
		return mean_logpdf(x, self.sigma, self.mu, mean_lnx2)

	@staticmethod
	def fit(x, xmin=None, xmax=None):
		return fit_mle(x, xmin=xmin, xmax=xmax)
