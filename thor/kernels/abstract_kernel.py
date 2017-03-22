import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.spatial.distance import cdist


class AbstractKernel(object):
    """Abstract Kernel Class"""
    __metaclass__ = ABCMeta

    def __init__(self, amplitude, length_scales, noise_level):
        """Initialize parameters of the abstract kernel object."""
        self.amplitude = amplitude
        self.length_scales = length_scales
        self.noise_level = noise_level

    def __str__(self):
        s = "{}\n".format(self.kernel_name)
        s += "\tAmplitude:\t{:.4f}\n".format(self.amplitude)
        s += "\tLength Scales:\n"
        for ls in self.length_scales:
            s += "\t\t\t{:.4f}\n".format(ls)
        s += "\tNoise Level:\t{:.4f}".format(self.noise_level)
        return s

    def pairwise_distances(self, X, Y):
        """Computes the pairwise distances of all of the rows in one matrix to
        all of the rows of the other.
        """
        # Rescale the inputs using the length scales.
        Xp = X / self.length_scales
        Yp = Y / self.length_scales

        return cdist(Xp, Yp, "sqeuclidean")

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
