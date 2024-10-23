import math
import numpy as np
from scipy.optimize import minimize
from .mle import MLEModification


def pareto(x, a, x_min):
	return ((a - 1) / x_min) * ((x / x_min) ** (-a))


def _a(alpha):
	return ((alpha - 1) * np.exp(alpha) + 1) / (alpha * (alpha - 1))


def pdf(x: np.ndarray, alpha, scale):
	c = 1 / (scale * _a(alpha))
	y = x.copy()
	mask1 = x >= scale
	mask2 = x < scale
	y[mask1] = (x[mask1] / scale) ** (-alpha)
	y[mask2] = (np.exp((-alpha) * (x[mask2] / scale - 1)))
	return c * y


def cdf(x: np.ndarray, alpha, scale):
	a_inv = 1 / _a(alpha)
	x = np.array(x)
	y = np.empty(x.shape)
	mask1 = x >= scale
	mask2 = x < scale
	y[mask1] = 1 - (a_inv / (alpha - 1)) * (x[mask1] / scale) ** (1 - alpha)
	y[mask2] = a_inv / alpha * np.exp(alpha) * (1 - np.exp(-alpha * (x[mask2] / scale)))
	return y


def mean_logpdf(x: np.ndarray, alpha, scale):
	x_part1 = x[x < scale]
	x_part2 = x[x >= scale]
	n = len(x)
	l1 = - np.log(scale) - np.log(_a(alpha))
	l2 = -alpha/n * np.sum(x_part1 / scale - 1) if len(x_part1) > 0 else alpha/n
	l3 = -alpha/n * np.sum(np.log(x_part2 / scale)) if len(x_part2) > 0 else 0
	return l1 + l2 + l3


def get_functional(x, xmin=None, xmax=None):
	modification = MLEModification(cdf, xmin, xmax)

	def functional(theta):
		l1 = -mean_logpdf(x, *theta)
		l2 = -modification(theta)
		return l1 + l2

	return functional


def fit_mle(x, xmin=None, xmax=None):
	alpha_0 = 1 + 1 / (np.mean(np.log(x)) - np.log(1))
	def functional(theta): return -mean_logpdf(x, theta[0], theta[1])
	theta_0 = np.array([alpha_0, 2])
	res = minimize(
		get_functional(x, xmin=xmin, xmax=xmax),
		theta_0,
		bounds=((1 + 1e-3, None), (1e-3, None)),
		method='Nelder-Mead',
		tol=1e-3
	)
	return res.x[0], res.x[1]


class paretoexp:
	def __init__(self, alpha, scale):
		self.alpha = alpha
		self.scale = scale

	def pdf(self, x):
		return pdf(x, self.alpha, self.scale)

	def cdf(self, x):
		return cdf(x, self.alpha, self.scale)

	def mean_logpdf(self, x):
		return mean_logpdf(x, self.alpha, self.scale)

	@staticmethod
	def fit(x, xmin=None, xmax=None):
		return fit_mle(x, xmin=xmin, xmax=xmax)
