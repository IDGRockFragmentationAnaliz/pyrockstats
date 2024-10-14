import math
import numpy as np
from scipy.special import erf
from scipy.special import gammainc
from scipy.special import gamma
from scipy.optimize import minimize


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


def fit_mle(x, x_max=None, x_min=None):
	mean_lnx = np.mean(np.log(x))
	mean_lnx2 = np.mean(np.log(x) ** 2)

	def functional(theta):
		return -mean_logpdf(
			x, theta[0], theta[1],
			mean_lnx=mean_lnx,
			mean_lnx2=mean_lnx2)

	theta_0 = np.array([1, 1])
	res = minimize(
		functional,
		theta_0,
		bounds=((1e-3, None), (1e-3, None)),
		method='nelder-mead',
		tol=1e-3
	)

	return res.x[0], res.x[1]


class lognorm:
	def __init__(self, sigma, scale):
		self.sigma = sigma
		self.scale = scale
		self.mu = math.log(scale)

	def pdf(self, x):
		return pdf(x, self.sigma, self.scale)

	def cdf(self, x):
		return cdf(x, self.sigma, self.scale)

	def mean_logpdf(self, x, mean_lnx2):
		return mean_logpdf(x, self.sigma, self.mu, mean_lnx2)

	@staticmethod
	def fit(x, x_max=None, x_min=None):
		return fit_mle(x, x_max, x_min)
