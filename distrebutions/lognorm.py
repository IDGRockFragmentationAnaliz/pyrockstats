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
	def __init__(self, sigma, scale, xmin=None, xmax=None):
		self.xmin = xmin if xmin is not None else None
		self.xmax = xmax if xmax is not None else None
		self.sigma = sigma
		self.scale = scale
		self.mu = math.log(scale)

	def pdf(self, x, xmin=None, xmax=None):
		if xmin is None and xmax is None:
			return pdf(x, self.sigma, self.scale)
		cdf_min = cdf(xmin, self.sigma, self.scale) \
			if xmin is not None else 0
		cdf_max = cdf(xmax, self.sigma, self.scale) \
			if xmax is not None else 1
		
		return pdf(x, self.sigma, self.scale)/(cdf_max - cdf_min)

	def cdf(self, x, xmin=None, xmax=None):
		if xmin is None and xmax is None:
			return cdf(x, self.sigma, self.scale)
		cdf_min = cdf(xmin, self.sigma, self.scale) \
			if xmin is not None else 0
		cdf_max = cdf(xmax, self.sigma, self.scale) \
			if xmax is not None else 1
		
		return (cdf(x, self.sigma, self.scale) - cdf_min)/(cdf_max - cdf_min)
	
	def rvs(self, n, xmin=None, xmax=None):
		rv = self.sigma * np.random.normal(size=n) + self.mu
		if xmin is None and xmax is None:
			return np.exp(rv)
		logxmin = np.log(xmin) if xmin is not None or xmin != 0 else -np.inf
		logxmax = np.log(xmax) if xmax is not None else np.inf
		mask = np.where(np.logical_or(rv < logxmin, rv > logxmax))[0]
		n = mask.shape[0]
		while n > 0:
			rv[mask] = self.sigma * np.random.normal(size=n) + self.mu
			idxs = np.where(np.logical_and(rv[mask] < logxmin, rv[mask] > logxmax))[0]
			mask = mask[idxs]
			n = mask.shape[0]
		return np.exp(rv)
	
	def mean_logpdf(self, x, mean_lnx2):
		return mean_logpdf(x, self.sigma, self.mu, mean_lnx2)

	@staticmethod
	def fit(x, xmin=None, xmax=None):
		return fit_mle(x, xmin=xmin, xmax=xmax)
