import numpy as np
from abc import ABCMeta, abstractmethod


class Dimension(object):
    """Bayesian Optimization Input Dimension Class

    Bayesian optimization seeks to identify to the maxima of a latent objective
    function by carefully tuning each dimension. This class, therefore,
    implements the dimension object that specifies how to map the dimension into
    the unit interval, how to invert a point in the unit interval to the
    original space, and how to sample from the dimension.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform(self, original_input):
        """Transforms an original input object to a real number in the unit
        interval.

        Parameters:
            original_input (int or float): A point in the original dimension.

        Returns:
            A point in the unit interval that corresponds to an input in the
                space of the original dimension. Note that for certain dimension
                types (namely integer dimensions) the transformation function
                may not be one-to-one.

        Raises:
            NotImplementedError: This is an abstract method that is not
                implemented in the abstract base class.
        """
        raise NotImplementedError()

    @abstractmethod
    def invert(self, transformed_input):
        """Inverts a transformed input real number in the unit interval to an
        object from the original dimension.

        Parameters:
            transformed_input (float): A point in the unit interval (inclusive)
                that can be transformed back into the space of the dimension by
                applying this function.

        Returns:
            A point in the space of the original dimension that corresponds to
                an input in the unit interval. Note that for certain dimension
                types (namely integer dimensions) the inversion function may not
                be one-to-one.

        Raises:
            NotImplementedError: This is an abstract method that is not
                implemented in the abstract base class.
        """
        raise NotImplementedError()

    def sample(self):
        """Sample from the original dimension by sampling from a uniform
        distribution (the transformed space) and mapping it back to the original
        dimension.

        Returns:
            A point uniformly selected from the dimension. This is computed by
                randomly sampling from the unit interval and then transforming
                the point back to the space of the original dimension.
        """
        u = np.asarray(np.random.uniform())
        return self.invert(u)


class LinearDimension(Dimension):
    """Bayesian Optimization Linear Dimension Class

    This class implements a dimension of uniformly spaced points on a continuum
    between a low point and a high point.

    Parameters:
        low (float): The minimum of the dimension.
        high (float): The maximum of the dimension.

    Raises:
        ValueError: If low is greater than or equal to the high, then a value
            error is produced.
    """
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


class LogarithmicDimension(LinearDimension):
    """Bayesian Optimization Logarithmic Dimension Class"""
    def __init__(self, low, high):
        super(LogarithmicDimension, self).__init__(np.log(low), np.log(high))

    def transform(self, original_input):
        """Extension of base class method."""
        return super(LogarithmicDimension, self).transform(np.log(original_input))

    def invert(self, transformed_input):
        """Extension of base class method."""
        return np.exp(super(LogarithmicDimension, self).invert(transformed_input))


class ExponentialDimension(LinearDimension):
    """Bayesian Optimization Exponential Dimension Class"""
    def __init__(self, low, high):
        super(ExponentialDimension, self).__init__(np.exp(low), np.exp(high))

    def transform(self, original_input):
        """Extension of base class method."""
        return super(ExponentialDimension, self).transform(np.exp(original_input))

    def invert(self, transformed_input):
        """Extension of base class method."""
        return np.log(super(ExponentialDimension, self).invert(transformed_input))


class IntegerDimension(LinearDimension):
    """Bayesian Optimization Integer Dimension Class

    This class implements a dimension that only takes integer values between a
    specified low and high value.
    """
    def __init__(self, low, high):
        """Initialize the parameters of the integer dimension class."""
        super(IntegerDimension, self).__init__(int(low), int(high))
        self.integers = np.array([i for i in range(self.low, self.high+1)])
        self.n_integers = len(self.integers)
        self.delta = np.linspace(0., 1., num=self.n_integers + 1)

    def invert(self, transformed_input):
        """Implementation of abstract base class method."""
        for i in range(self.n_integers):
            if (
                    transformed_input >= self.delta[i] and
                    transformed_input <= self.delta[i + 1]
            ):
                return self.integers[i]

