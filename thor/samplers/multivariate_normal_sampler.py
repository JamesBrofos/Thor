import numpy as np
import scipy.linalg as spla


class MultivariateNormalSampler(object):
    """Numerically Stable Multivariate Normal Sampler Class"""
    def __init__(self, n_sample, mean, cov):
        """Initialize the parameters of the multivariate normal sampler object.
        """
        self.n_sample = n_sample
        self.mean = mean
        self.cov = cov

    def sample(self):
        k = self.cov.shape[0]
        L = spla.cholesky(self.cov, True)
        return self.mean + L.dot(np.random.normal(size=(k, self.n_sample))).T

