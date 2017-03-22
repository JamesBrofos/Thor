import numpy as np
from abc import ABCMeta, abstractmethod


class Dimension(object):
    """Bayesian Optimization Input Dimension Class"""
    __metaclass__ = ABCMeta

    # def __init__(self, name):
    #     """Initialize parameters of the input dimension object."""
    #     self.name = name

    @abstractmethod
    def transform(self, original_input):
        """Transforms an original input object to a real number in the unit
        interval.
        """
        raise NotImplementedError()

    @abstractmethod
    def invert(self, transformed_input):
        """Inverts a transformed input real number in the unit interval to an
        object from the original dimension.
        """
        raise NotImplementedError()

    def sample(self):
        """Sample from the original dimension by sampling from a uniform
        distribution (the transformed space) and mapping it back to the original
        dimension.
        """
        u = np.asarray(np.random.uniform())
        return self.invert(u)


class LinearDimension(Dimension):
    """Bayesian Optimization Linear Dimension Class"""
    def __init__(self, low, high):
        """Initialize parameters of the linear dimension object."""
        self.low, self.high = low, high
        self.__diff = self.high - self.low
        if self.low >= self.high:
            raise ValueError(
                "Invalid ordering of low and high dimensions:\t{} >= {}".format(
                    self.low, self.high
                )
            )

    def transform(self, original_input):
        """Implementation of abstract base class method."""
        return (original_input - self.low) / self.__diff

    def invert(self, transformed_input):
        """Implementation of abstract base class method."""
        return transformed_input * self.__diff + self.low


class IntegerDimension(LinearDimension):
    """Bayesian Optimization Integer Dimension Class
    """
    def __init__(self, low, high):
        """Initialize the parameters of the integer dimension class."""
        if type(low) != int or type(high) != int:
            raise ValueError("Invalid bounds parameter in integer dimension.")
        super(IntegerDimension, self).__init__(low, high)
        self.integers = np.array([i for i in range(low, high+1)])
        self.n_integers = len(self.integers)
        self.delta = np.linspace(0., 1., num=self.n_integers + 1)

    def invert(self, transformed_input):
        """Implementation of abstract base class method."""
        for i in range(self.n_integers):
            if (
                    transformed_input > self.delta[i] and
                    transformed_input <= self.delta[i + 1]
            ):
                return self.integers[i]



