from abc import ABC, abstractmethod


class distribution_continuous(ABC):
	@abstractmethod
	def cdf(self, x):
		pass

	@abstractmethod
	def pdf(self, y):
		pass

	@abstractstaticmethod
	def fit(s):
		pass
