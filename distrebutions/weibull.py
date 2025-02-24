import math
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
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
		l_0 = -mean_logpdf(x, *theta, mean_lnx=np.mean(np.log(x)))
		return l_0 + modification(theta)

	return functional


def mle_alpha_equation(alpha, x):
	m_x = np.mean(x ** alpha)
	m_lx = np.mean(np.log(x))
	m_xlx = np.mean(x ** alpha * np.log(x))
	den = m_xlx - m_x * m_lx
	return alpha * den - m_x


def fit_mle(x, xmax=None, xmin=None):
	mx = np.mean(x)
	x = x / mx
	xmin = xmin / mx if xmin is not None else xmin
	xmax = xmax / mx if xmax is not None else xmax
	
	alpha_0 = 1
	lambda_0 = np.mean(x)
	
	res = fsolve(
		mle_alpha_equation, np.array(alpha_0), args=(x,))
	
	alpha_0 = res[0]
	lambda_0 = np.mean(x ** alpha_0) ** (1/alpha_0)
	
	theta_0 = np.array([alpha_0 + 2e-3, lambda_0])
	print("initial", theta_0)
	res = minimize(
		get_functional(x, xmin=xmin, xmax=xmax),
		theta_0,
		bounds=((1e-3, None), (1e-3, None)),
		method='Nelder-Mead', tol=1e-3
	)
	print("final",[res.x[0], res.x[1]])
	
	return res.x[0], res.x[1]*mx


class weibull:
	def __init__(self, alpha, scale):
		self.alpha = alpha
		self.scale = scale

	def pdf(self, x, xmin=None, xmax=None):
		if xmin is None and xmax is None:
			return pdf(x, self.alpha, self.scale)
		cdf_min = cdf(xmin, self.alpha, self.scale) \
			if xmin is not None else 0
		cdf_max = cdf(xmax, self.alpha, self.scale) \
			if xmax is not None else 1
		return pdf(x, self.alpha, self.scale) / (cdf_max - cdf_min)

	def cdf(self, x, xmin=None, xmax=None):
		if xmin is None and xmax is None:
			return cdf(x, self.alpha, self.scale)
		cdf_min = cdf(xmin, self.alpha, self.scale) \
			if xmin is not None else 0
		cdf_max = cdf(xmax, self.alpha, self.scale) \
			if xmax is not None else 1
		return (cdf(x, self.alpha, self.scale) - cdf_min) / (cdf_max - cdf_min)

	def mean_logpdf(self, x, mean_lnx=None):
		return mean_logpdf(x, self.alpha, self.scale, mean_lnx)

	@staticmethod
	def fit(x, xmin=None, xmax=None):
		return fit_mle(x, xmin=xmin, xmax=xmax)

