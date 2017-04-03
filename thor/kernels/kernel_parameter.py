import numpy as np


class KernelParameter(object):
    """Kernel Parameter Class"""
    def __init__(self, name, value, bounds):
        """Initialize parameters of the kernel parameter object."""
        self.name = name
        self.value = np.atleast_1d(value)
        self.bounds = bounds

    def sample(self):
        """Sample from a uniform distribution in the interval defined by the
        bounds of the kernel parameter.
        """
        return np.random.uniform(
            low=self.bounds[0], high=self.bounds[1], size=self.value.shape
        )
