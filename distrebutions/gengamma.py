import math
import numpy as np
from scipy.special import erf
from scipy.special import gammainc
from scipy.special import gamma
from scipy.optimize import minimize

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


def fit_mle(x):
	mean_lnx = np.mean(np.log(x))

	def functional(_theta):
		return -mean_logpdf(x, _theta[0], _theta[1], _theta[2], mean_lnx=mean_lnx)

	theta_0 = np.array([1, 1, 1])
	theta = minimize(
		functional,
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
	def fit(x):
		return fit_mle(x)
