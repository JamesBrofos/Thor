import numpy as np
from .abstract_kernel import AbstractKernel


class SquaredExponentialKernel(AbstractKernel):
    """Squared Exponential Kernel Class"""
    def __init__(self, amplitude, length_scales, noise_level):
        """Initialize parameters of the squared exponential kernel object."""
        super(SquaredExponentialKernel, self).__init__(
            amplitude, length_scales, noise_level
        )
        self.kernel_name = "Squared Exponential"

    def cov(self, X, Y=None):
        """Implementation of abstract base class method."""
        # Handle the case where we are computing the regular covariance instead
        # of the cross covariance.
        if Y is None:
            Y = X
            noise_variance = self.noise_level * np.eye(X.shape[0])
        else:
            noise_variance = 0.
        # Compute the squared distances between inputs under the current length
        # scales.
        r_sq = self.pairwise_distances(X, Y)
        # Compute the squared exponential kernel.
        K = self.amplitude * np.exp(-0.5 * r_sq) + noise_variance + 1e-5

        return K

    def grad_input(self, x, Y):
        """Implementation of abstract base class method."""
        d_dist = -(x - Y) / (self.length_scales ** 2)
        d_kernel = self.cov(np.atleast_2d(x), Y).T
        grad = d_dist * d_kernel
        return grad

    def grad_params(self, X):
        """Implementation of abstract base class method."""
        K_amp_grad = self.cov(X) - self.noise_level * np.eye(X.shape[0])
        K_ls_grad = ((
            (np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0)) ** 2 /
            (self.length_scales ** 3)
        ) * K_amp_grad[..., np.newaxis])
        return K_amp_grad, K_ls_grad
