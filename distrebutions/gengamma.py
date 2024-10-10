import math
import numpy as np
from scipy.special import erf
from scipy.special import gammainc
from scipy.special import gamma


def cdf_gengamma(x, a, b, lam):
	return gammainc(a, np.power(x / lam, b))


def pdf_gengamma(x, a, b, lam):
	return b / lam / gamma(a) * np.power(x / lam, a * b - 1) * np.exp(-np.power(x / lam, b))


class Gengamma:
	def __init__(self, a, b, lam):
		self.a = a
		self.b = b
		self.lam = lam

	def pdf(self, x):
		return pdf_gengamma(x, self.a, self.b, self.lam)

	def cdf(self, x):
		return cdf_gengamma(x, self.a, self.b, self.lam)

	def mean_logpdf(self, x, mean_lnx=None):
		mean_lnx = mean_lnx if mean_lnx is not None else np.mean(np.log(x))

		l1 = -math.log(gamma(self.a)) + math.log(self.b) - math.log(self.lam)
		l2 = (self.a * self.b - 1) * (mean_lnx - math.log(self.lam))
		l3 = -np.mean(np.power(x / self.lam, self.b))
		return l1 + l2 + l3
