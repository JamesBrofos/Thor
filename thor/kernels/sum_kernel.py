from functools import reduce
from .abstract_kernel import AbstractKernel


class SumKernel(AbstractKernel):
    """Sum Kernel Class"""
    def __init__(self, kernels, noise_kernel):
        """Initialize parameters of the sum kernel object."""
        self.kernels = kernels
        self.noise_kernel = noise_kernel

    @property
    def parameters(self):
        return reduce(
            lambda a, b: a+b, [k.parameters for k in self.kernels]
        ) + self.noise_kernel.parameters

    def cov(self, X, Y=None, include_noise=True):
        """Implementation of abstract base class method."""
        K = reduce(
            lambda K, H: K+H, [k.cov(X, Y) for k in self.kernels]
        )
        if include_noise:
            return K + self.noise_kernel.cov(X, Y)
        else:
            return K

    def grad_input(self, x, Y):
        """Implementation of abstract base class method."""
        return reduce(
            lambda K, H: K+H, [k.grad_input(x, Y) for k in self.kernels]
        )

    def grad_params(self, X):
        """Implementation of abstract base class method."""
        grads = {
            p: v for k in self.kernels for p, v in k.grad_params(X).items()
        }
        for p, v in self.noise_kernel.grad_params(X).items():
            grads[p] = v

        return grads
