import numpy as np
from .abstract_kernel import AbstractKernel
from .kernel_parameter import KernelParameter


class NoiseKernel(AbstractKernel):
    """Noise Kernel Class"""
    def __init__(self, noise):
        """Initialize parameters of the noise kernel object."""
        self.noise = KernelParameter("noise", noise, (1e-3, 1e-1))

    @property
    def parameters(self):
        return (self.noise, )

    def cov(self, X, Y=None):
        """Implementation of abstract base class method."""
        if Y is None:
            return self.noise.value * np.eye(X.shape[0])
        else:
            return 0.

    def grad_input(self, x, Y):
        """Implementation of abstract base class method."""
        return 0.

    def grad_params(self, X):
        """Implementation of abstract bsae class method."""
        return {self.noise.name: np.eye(X.shape[0])}
