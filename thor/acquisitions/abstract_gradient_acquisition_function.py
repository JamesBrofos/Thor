import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from abc import ABCMeta, abstractmethod
from .abstract_acquisition_function import AbstractAcquisitionFunction


class AbstractGradientAcquisitionFunction(AbstractAcquisitionFunction):
    """Abstract Gradient-Based Acquisition Function"""
    __metaclass__ = ABCMeta

    def __negative_acquisition_function(self, params):
        return -self.evaluate(np.atleast_2d(params))

    def __negative_acquisition_function_grad(self, params):
        return -self.grad_input(np.atleast_2d(params))

    def __maximize(self):
        """Helper function that leverages the BFGS algorithm with a bounded
        input space in order to converge the maximum of the acquisition function
        using gradient ascent. Notice that this function returns the point in
        the original input space, not the point in the unit hypercube.
        """
        k = self.model.X.shape[1]
        bounds = [(0., 1.)] * k
        x = np.random.uniform(low=0., high=1., size=(1, k))
        res = fmin_l_bfgs_b(
            self.__negative_acquisition_function,
            x,
            fprime=self.__negative_acquisition_function_grad,
            bounds=bounds,
            disp=0
        )
        x_max = np.atleast_2d(res[0])
        return x_max.ravel(), self.evaluate(x_max)

    def maximize(self, n_evals):
        """Implementation of abstract base class method."""
        best_acq = -np.inf
        for i in range(n_evals):
            x, val = self.__maximize()
            if val > best_acq:
                x_best = x
                best_acq = val

        return x_best

    @abstractmethod
    def evaluate(self, X):
        """Evaluate the acquisition function at the specified inputs."""
        raise NotImplementedError()

    @abstractmethod
    def grad_input(self, x):
        """Compute the gradient of the acquisition function with respect to the
        inputs.
        """
        raise NotImplementedError()
