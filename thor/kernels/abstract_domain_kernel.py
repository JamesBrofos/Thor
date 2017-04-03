import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.spatial.distance import cdist
from .kernel_parameter import KernelParameter
from .abstract_kernel import AbstractKernel


class AbstractDomainKernel(AbstractKernel):
    """Abstract Domain Kernel Class"""
    __metaclass__ = ABCMeta

    def __init__(self, amplitude, length_scales):
        """Initialize parameters of the abstract domain kernel object."""
        self.amplitude = KernelParameter("amplitude", amplitude, (1e-3, 2.))
        self.length_scales = KernelParameter(
            "length_scales", length_scales, (1e-2, 1.)
        )

    def pairwise_distances(self, X, Y):
        """Computes the pairwise distances of all of the rows in one matrix to
        all of the rows of the other.
        """
        # Rescale the inputs using the length scales.
        Xp = X / np.sqrt(self.length_scales.value)
        Yp = Y / np.sqrt(self.length_scales.value)

        return cdist(Xp, Yp, "sqeuclidean")

    @property
    def parameters(self):
        """Implementation of abstract base class property."""
        return (self.amplitude, self.length_scales)
