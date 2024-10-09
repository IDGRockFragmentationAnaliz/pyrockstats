import math
import numpy as np
from scipy.special import erf
from scipy.special import gammainc
from scipy.special import gamma


def pdf_lognorm(x, sigma, scale):
	return (1 / (x * sigma * np.sqrt(2 * math.pi))) * np.exp(-0.5 * ((np.log(x / scale)) / sigma) ** 2)


def cdf_lognorm(x, sigma, scale):
	return 0.5 * (1 + erf(np.log(x / scale) / (sigma * math.sqrt(2))))


class Lognorm:
	def __init__(self, sigma, scale):
		self.sigma = sigma
		self.scale = scale

	def pdf(self, x):
		return pdf_lognorm(x, self.sigma, self.scale)

	def cdf(self, x):
		return cdf_lognorm(x, self.sigma, self.scale)
