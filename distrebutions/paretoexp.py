import math
import numpy as np


def pareto(x, a, x_min):
	return ((a - 1) / x_min) * ((x / x_min) ** (-a))


def _a(alpha):
	return ((alpha - 1) * np.exp(alpha) + 1) / (alpha * (alpha - 1))


def pdf_paretoexp(x, alpha, scale):
	c = 1 / (scale * _a(alpha))
	y = x.copy()
	mask1 = x >= scale
	mask2 = x < scale
	y[mask1] = (x[mask1] / scale) ** (-alpha)
	y[mask2] = (np.exp((-alpha) * (x[mask2] / scale - 1)))
	return c * y


def cdf_paretoexp(x: np.ndarray, alpha, scale):
	a_inv = 1 / _a(alpha)
	x = np.array(x)
	y = np.empty(x.shape)
	mask1 = x >= scale
	mask2 = x < scale
	y[mask1] = 1 - (a_inv / (alpha - 1)) * (x[mask1] / scale) ** (1 - alpha)
	y[mask2] = a_inv / alpha * np.exp(alpha) * (1 - np.exp(-alpha * (x[mask2] / scale)))
	return y


class Paretoexp:
	def __init__(self, alpha, scale):
		self.alpha = alpha
		self.scale = scale

	def pdf(self, x):
		return pdf_paretoexp(x, self.alpha, self.scale)

	def cdf(self, x):
		return cdf_paretoexp(x, self.alpha, self.scale)
