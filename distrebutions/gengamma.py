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

	def meanlogpdf(self, x, MlnX):
		W1 = -math.log(gamma(self.a)) + math.log(self.b) - math.log(self.lam)
		W2 = (self.a * self.b - 1) * (MlnX - math.log(self.lam))
		W3 = -np.mean(np.power(x / self.lam, self.b))
		return W1 + W2 + W3
