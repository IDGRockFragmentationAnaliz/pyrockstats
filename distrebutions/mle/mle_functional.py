import numpy as np
from .mle_modification import MLEModification


class MLEFunctional:
	def __init__(self, x, cdf=None, xmin=None, xmax=None):
		self.modification = MLEModification(cdf, xmin, xmax)
