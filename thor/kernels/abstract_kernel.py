from abc import ABCMeta, abstractmethod, abstractproperty
from functools import reduce


class AbstractKernel(object):
    """Abstract Kernel Class"""
    __metaclass__ = ABCMeta

    def sample(self):
        """Sample kernel parameters."""
        return {p.name: p.sample() for p in self.parameters}

    def update(self, values):
        """Update the parameters of the kernel using provided values."""
        idx = 0
        for p in self.parameters:
            sz = len(p.value)
            p.value = values[idx:(idx+sz)]
            idx += sz

    @property
    def bounds(self):
        """Retrieve the bounds of the kernel parameters."""
        bnds = []
        for p in self.parameters:
            for _ in range(len(p.value)):
                bnds.append(p.bounds)

        return bnds

    @abstractproperty
    def parameters(self):
        """Returns the parameters of the kernel, including such values as the
        length scales, the amplitude, and the noise.
        """
        raise NotImplementedError()

    @abstractmethod
    def cov(self, X, Y=None):
        """Any kernel must implement a covariance (or, indeed, a cross
        covariance) function between inputs. Notice that this is independent of
        the outputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def grad_input(self, x, Y):
        """Compute the gradient of the kernel with respect to the input."""
        raise NotImplementedError()

    @abstractmethod
    def grad_params(self, X):
        """Compute the gradient of the kernel with respect to the kernel
        parameters: the amplitude, the length scales, and the noise variance.

        This function will return gradients with respect the kernel parameters
        in the order above.
        """
        raise NotImplementedError()
