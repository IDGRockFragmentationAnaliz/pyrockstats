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


def mean_logpdf(x, sigma, mu, mean_lnx2=None):
	mean_lnx2 = mean_lnx2 if mean_lnx2 is not None else np.mean(np.log(x ** 2))

	part1 = -np.log(sigma) - mean_lnx2 / (2 * (sigma ** 2))
	part2 = (2 * mu * mean_lnx2 - mu ** 2) / (2 * (sigma ** 2))
	return part1 + part2


def fit_mle(x, x_max=None, x_min=None):
	mean_lnx = np.mean(np.log(x))
	functional = lambda theta: -mean_logpdf(x, theta[0], theta[1], mean_lnx)
	theta_0 = [1, 0]
	res = minimize(
		functional,
		theta_0,
		bounds=((1e-3, None), (-np.inf, np.inf)),
		method='nelder-mead',
		tol=1e-3
	)
	return res.x[0], res.x[1]


class Lognorm:
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
