import math
import numpy as np
from scipy.special import erf
from scipy.special import gammainc
from scipy.special import gamma
from scipy.optimize import minimize
from .mle import MLEModification


def cdf(x, a, b, lam):
	return gammainc(a, np.power(x / lam, b))


def pdf(x, a, b, lam):
	return b / lam / gamma(a) * np.power(x / lam, a * b - 1) * np.exp(-np.power(x / lam, b))


def mean_logpdf(x, a, b, scale, mean_lnx=None):
	mean_lnx = np.mean(np.log(x)) if mean_lnx is None else mean_lnx
	l1 = -math.log(gamma(a)) + math.log(b) - math.log(scale)
	l2 = (a * b - 1) * (mean_lnx - math.log(scale))
	l3 = -np.mean(np.power(x / scale, b))
	return l1 + l2 + l3


def functional(theta, x, xmin=None, xmax=None, mean_lnx=None):
	l1 = mean_logpdf(x, *theta, mean_lnx=mean_lnx)
	l2 = (MLEModification(cdf, xmin, xmax))(theta)
	return -(l1 + l2)


def fit_mle(x, xmin=None, xmax=None):
	mean_lnx = np.mean(np.log(x))
	theta_0 = np.array([1, 1, 1])
	theta = minimize(
		lambda _theta: functional(_theta, x, xmin=xmin, xmax=xmax, mean_lnx=mean_lnx),
		theta_0,
		bounds=((1e-3, None), (1e-3, None), (1e-3, None)),
		method='Nelder-Mead', tol=1e-3
	)["x"]
	return theta[0], theta[1], theta[2]


class gengamma:
	def __init__(self, a, b, scale):
		self.a = a
		self.b = b
		self.scale = scale

	def pdf(self, x):
		return pdf(x, self.a, self.b, self.scale)

	def cdf(self, x):
		return cdf(x, self.a, self.b, self.scale)

	@staticmethod
	def fit(x, xmin=None, xmax=None):
		return fit_mle(x, xmin=xmin, xmax=xmax)
