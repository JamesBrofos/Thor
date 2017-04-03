import numpy as np
from .abstract_domain_kernel import AbstractDomainKernel


class SquaredExponentialKernel(AbstractDomainKernel):
    """Squared Exponential Kernel Class"""
    def cov(self, X, Y=None):
        """Implementation of abstract base class method."""
        # Handle the case where we are computing the regular covariance instead
        # of the cross covariance.
        if Y is None:
            Y = X
        # Compute the squared distances between inputs under the current length
        # scales.
        r_sq = self.pairwise_distances(X, Y)
        # Compute the squared exponential kernel.
        K = self.amplitude.value * np.exp(-0.5 * r_sq)

        return K

    def grad_input(self, x, Y):
        """Implementation of abstract base class method."""
        d_dist = -(x - Y) / (self.length_scales.value ** 2)
        d_kernel = self.cov(np.atleast_2d(x), Y).T
        grad = d_dist * d_kernel

        return grad

    def grad_params(self, X):
        """Implementation of abstract base class method."""
        K = self.cov(X)
        K_amp_grad = K / self.amplitude.value
        K_ls_grad = (
            (np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0)) ** 2 /
            (self.length_scales.value ** 2)
        ) * K[..., np.newaxis]

        return {
            self.amplitude.name: K_amp_grad,
            self.length_scales.name: K_ls_grad
        }
