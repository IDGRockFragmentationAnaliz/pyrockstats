import math
import numpy as np


def pdf_weibull(x, alpha, scale):
	return (alpha / scale) * ((x / scale) ** (alpha - 1)) * np.exp(-(x / scale) ** alpha)


def cdf_weibull(x, alpha, scale):
	return 1 - np.exp(-(x / scale) ** alpha)


class Weibull:
	def __init__(self, alpha, scale):
		self.alpha = alpha
		self.scale = scale

	def pdf(self, x):
		return pdf_weibull(x, self.alpha, self.scale)

	def cdf(self, x):
		return cdf_weibull(x, self.alpha, self.scale)
