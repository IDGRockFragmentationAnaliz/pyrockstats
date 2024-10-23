import math
import numpy as np
from scipy.optimize import minimize
from .mle import MLEModification


def pdf(x, alpha, scale):
	return (alpha / scale) * ((x / scale) ** (alpha - 1)) * np.exp(-(x / scale) ** alpha)


def cdf(x, alpha, scale):
	return 1 - np.exp(-(x / scale) ** alpha)


def mean_logpdf(x, alpha, scale, mean_lnx=None):
	mean_lnx = mean_lnx if mean_lnx is not None else np.mean(np.log(x))

	part1 = np.log(alpha / scale)
	part2 = (alpha - 1) * (mean_lnx - np.log(scale))
	part3 = -np.mean((x / scale) ** alpha)
	return part1 + part2 + part3


def get_functional(x, xmin=None, xmax=None):
	modification = MLEModification(cdf, xmin, xmax)

	def functional(theta):
		l1 = -mean_logpdf(x, *theta, mean_lnx=np.mean(np.log(x)))
		l2 = -modification(theta)
		return l1 + l2

	return functional


def fit_mle(x, xmax=None, xmin=None):
	mean_lnx = np.mean(np.log(x))
	theta_0 = np.array([1 + 2e-3, 1])
	res = minimize(
		get_functional(x, xmin=xmin, xmax=xmax),
		theta_0,
		bounds=((1e-3, None), (1e-3, None)),
		method='Nelder-Mead', tol=1e-3
	)
	return res.x[0], res.x[1]


class weibull:
	def __init__(self, alpha, scale):
		self.alpha = alpha
		self.scale = scale

	def pdf(self, x):
		return pdf(x, self.alpha, self.scale)

	def cdf(self, x):
		return cdf(x, self.alpha, self.scale)

	def mean_logpdf(self, x, mean_lnx=None):
		return mean_logpdf(x, self.alpha, self.scale, mean_lnx)

	@staticmethod
	def fit(x, xmin=None, xmax=None):
		return fit_mle(x, xmin=xmin, xmax=xmax)

