import numpy as np
import scipy.linalg as spla
from .abstract_process import AbstractProcess


class GaussianProcess(AbstractProcess):
    """Gaussian Process Class"""
    @property
    def parameters(self):
        """Implementation of abstract base class property."""
        return self.kernel.parameters

    @property
    def grad_prefactor(self):
        """Implementation of abstract base class property."""
        return 1.

    @property
    def predict_prefactor(self):
        """Implementation of abstract base class property."""
        return 1.

    def sample(self, X):
        """Extension of base class method."""
        # Get the mean and covariance.
        mean, cov = super(GaussianProcess, self).sample(X)
        return np.random.multivariate_normal(mean, cov)

    def log_likelihood(self):
        """Compute the log-likelihood of the data under the Gaussian process
        model with the given length scales, amplitude, and noise level of the
        kernel.
        """
        n = self.X.shape[0]
        return -1. * (
            0.5 * self.beta +
            np.sum(np.log(np.diag(self.L))) +
            0.5 * n * np.log(2.*np.pi)
        ) / n
