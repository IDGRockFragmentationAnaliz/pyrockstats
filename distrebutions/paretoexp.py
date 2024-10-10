import math
import numpy as np
from scipy.optimize import minimize

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
	part1 = - np.log(scale) - np.log(_a(alpha))
	part2 = -alpha * np.mean(x[x < scale] / scale - 1)
	part3 = -alpha * np.mean(np.log(x[x >= scale] / scale))
	print("part1", part1)
	print("part2", part2)
	print("part3", part3)
	return part1 + part2 + part3


def fit_mle(x):
	alpha_0 = 1 + 1 / (np.mean(np.log(x)) - np.log(1))
	functional = lambda theta: mean_logpdf(x, theta[0], theta[1])


	print(alpha_0)
	print(functional([alpha_0, 2]))
	return 1

	res = minimize(
		functional,
		[alpha_0, 2],
		bounds=((1 + 1e-3, None), (1e-3, None)),
		method='Nelder-Mead',
		tol=1e-3
	)
	return res.x[0], res.x[1]


class Paretoexp:
	def __init__(self, alpha, scale):
		self.alpha = alpha
		self.scale = scale

	def pdf(self, x):
		return pdf(x, self.alpha, self.scale)

	def cdf(self, x):
		return cdf(x, self.alpha, self.scale)

	def mean_logpdf(self, x):
		return mean_logpdf(x, self.alpha, self.scale)
