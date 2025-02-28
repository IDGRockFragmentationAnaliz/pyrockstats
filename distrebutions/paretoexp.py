import math
import numpy as np
from scipy.optimize import minimize
from .mle import MLEModification


def pareto(x, a, x_min):
    return ((a - 1) / x_min) * ((x / x_min) ** (-a))


def _a(alpha):
    return ((alpha - 1) * np.exp(alpha) + 1) / (alpha * (alpha - 1))


def pdf(x: np.ndarray, alpha, scale):
    c = 1 / (scale * _a(alpha))
    y = x.copy()
    mask1 = x >= scale
    mask2 = x < scale
    y[mask1] = (x[mask1] / scale) ** (-alpha)
    y[mask2] = (np.exp((-alpha) * (x[mask2] / scale - 1)))
    return c * y


def cdf_inversed(y: np.ndarray, alpha, scale):
    y = np.array(y)
    cdf_scale = (alpha-1)*(np.exp(alpha) - 1)/((alpha - 1)*np.exp(alpha) + 1)
    y_mask1 = np.where(y < cdf_scale)
    y_mask2 = np.where(y >= cdf_scale)
    x = np.empty(y.shape)
    x[y_mask1] = -scale/alpha*np.log(1 - alpha * np.exp(-alpha) * _a(alpha) * y[y_mask1])
    x[y_mask2] = scale * ((_a(alpha) * (alpha - 1) * (1 - y[y_mask2])) ** (1 / (1 - alpha)))
    return x


def cdf(x: np.ndarray, alpha, scale):
    scale_c = 1 / _a(alpha)
    x = np.array(x)
    y = np.empty(x.shape)
    mask_1 = x < scale
    mask_2 = x >= scale
    y[mask_1] = scale_c / alpha * np.exp(alpha) * (1 - np.exp(-alpha * (x[mask_1] / scale)))
    y[mask_2] = 1 - (scale_c / (alpha - 1)) * (x[mask_2] / scale) ** (1 - alpha)
    return y


def mean_logpdf(x: np.ndarray, alpha, scale):
    x_part1 = x[x < scale]
    x_part2 = x[x >= scale]
    n = len(x)
    l1 = - np.log(scale) - np.log(_a(alpha))
    l2 = -alpha / n * np.sum(x_part1 / scale - 1) if len(x_part1) > 0 else alpha / n
    l3 = -alpha / n * np.sum(np.log(x_part2 / scale)) if len(x_part2) > 0 else 0
    return l1 + l2 + l3


def get_functional(x, xmin=None, xmax=None):
    modification = MLEModification(cdf, xmin, xmax)
    
    def functional(theta):
        l1 = -mean_logpdf(x, *theta)
        l2 = modification(theta)
        return l1 + l2
    
    return functional


def fit_mle(x: np.ndarray, xmin=None, xmax=None):
    mx = np.mean(x)
    x = x / mx
    xmin = xmin / mx if xmin is not None else xmin
    xmax = xmax / mx if xmax is not None else xmax
    
    alpha_0 = 1 + 1 / (np.mean(np.log(x / np.min(x))))
    
    theta_0 = np.array([alpha_0, 2])
    res = minimize(
        get_functional(x, xmin=xmin, xmax=xmax),
        theta_0,
        bounds=((1 + 1e-3, None), (1e-3, None)),
        method='Nelder-Mead',
        tol=1e-3
    )
    return res.x[0], res.x[1] * mx


class paretoexp:
    def __init__(self, alpha, scale):
        self.alpha = alpha
        self.scale = scale
        self.c = 1 / (scale * _a(alpha))
    
    def pdf(self, x, xmin=None, xmax=None):
        if xmin is None and xmax is None:
            return cdf(x, self.alpha, self.scale)
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
    
    def rvs(self, n: int, xmin=None, xmax=None):
        rv_cdf = np.random.rand(n)
        rv = cdf_inversed(rv_cdf, alpha=self.alpha, scale=self.scale)
        return rv
    
    # self.scale
    
    def mean_logpdf(self, x):
        return mean_logpdf(x, self.alpha, self.scale)
    
    @staticmethod
    def fit(x, xmin=None, xmax=None):
        return fit_mle(x, xmin=xmin, xmax=xmax)
