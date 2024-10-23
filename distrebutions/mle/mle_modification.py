import numpy as np


class MLEModification:
	def __init__(self, cdf=None, xmin=None, xmax=None):
		self.cdf_max = lambda theta: 1
		self.cdf_min = lambda theta: 0
		if xmax is not None and cdf is not None:
			self.cdf_max = lambda theta: cdf(xmax, *theta)
		if xmin is not None and cdf is not None:
			self.cdf_min = lambda theta: cdf(xmin, *theta)

	def __call__(self, theta):
		return self.evaluate(theta)

	def evaluate(self, theta):
		return -np.log(self.cdf_max(theta) - self.cdf_min(theta))
