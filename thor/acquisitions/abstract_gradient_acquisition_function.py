import numpy as np
import sobol_seq
from scipy.optimize import fmin_l_bfgs_b
from abc import ABCMeta, abstractmethod
from .abstract_acquisition_function import AbstractAcquisitionFunction


class AbstractGradientAcquisitionFunction(AbstractAcquisitionFunction):
    """Abstract Gradient-Based Acquisition Function

    Many acquisition functions utilized by Bayesian optimization are
    differentiable with respect to the input parameters. These gradients can be
    leveraged in order to find the maxima of the acquisition function. This will
    generally lead to better maxima being identified than if one were simply to
    sample randomly from the unit hypercube and compute the acquisition
    function. In order to identify the maxima of the acquisition function, this
    class utilizes the BFGS algorithm.

    Nonetheless, it is worth emphasizing that acquisition functions, even if
    they are differentiable, will still be multimodal. As a result, it is
    necessary to perform multiple random restarts from different locations
    within the unit hypercube.
    """
    __metaclass__ = ABCMeta

    def __negative_acquisition_function(self, params):
        """This function simply computes the negative of the acquisition
        function. This is required since the BFGS algorithm will seek to
        minimize the function rather than maximize it; therefore, the find the
        maxima of the acquisition function, in practice we find the minima of
        the negative of the acquisition function.

        Parameters:
            params (numpy array): An input from the unit hypercube at which the
                acquisition function will be computed.

        Returns:
            The negative of the acquisition function.
        """
        return -self.evaluate(np.atleast_2d(params))

    def __negative_acquisition_function_grad(self, params):
        """This function simply computes the negative of the gradient of the
        acquisition function.

        Parameters:
            params (numpy array): An input from the unit hypercube at which the
                gradient will be computed.

        Returns:
            The negative of the gradient of the acquisition function.
        """
        return -self.grad_input(np.atleast_2d(params))

    def __maximize(self, index):
        """Helper function that leverages the BFGS algorithm with a bounded
        input space in order to converge the maximum of the acquisition function
        using gradient ascent. Notice that this function returns the point in
        the original input space, not the point in the unit hypercube.

        Parameters:
            index (int): An integer that keeps track of the index in the Sobol
                sequence at which to generate a pseudo-random configuration of
                inputs in the unit hypercube. This is done in order to exploit
                the optimally uniform properties of the Sobol sequence.

        Returns:
            A tuple containing first the numpy array representing the input in
                the unit hypercube that minimizes the negative of the
                acquisition function (or, equivalently, maximizes the
                acquisition function) as well as the value of the acquisition
                function at the discovered maximum.
        """
        # Number of dimensions.
        k = self.model.X.shape[1]
        # x = np.random.uniform(low=0., high=1., size=(1, k))
        x = sobol_seq.i4_sobol(k, index+1)[0]
        try:
            # Bounds on the search space used by the BFGS algorithm.
            bounds = [(0., 1.)] * k
            # Call the BFGS algorithm to perform the maximization.
            res = fmin_l_bfgs_b(
                self.__negative_acquisition_function,
                x,
                fprime=self.__negative_acquisition_function_grad,
                bounds=bounds,
                disp=0
            )
            x_max = np.atleast_2d(res[0])
        except AttributeError:
            # In the case of a non-differentiable model, instead use the initial
            # random sample.
            x_max = x

        return x_max.ravel(), self.evaluate(x_max)

    def select(self):
        """Implementation of abstract base class method."""
        # Initialize the best acquisition value to negative infinity. This will
        # allow any fit of the data to be better.
        best_acq = -np.inf
        # Compute the number of evaluations to perform.
        n_evals = 10 * self.model.X.shape[1]
        # For the specified number of iterations, try to maximize the
        # acquisition function using random search or randomly initialized
        # gradient ascent.
        for i in range(n_evals):
            x, val = self.__maximize(i)
            # When a better maximizer of the acquisition function is found, make
            # note of it.
            if val > best_acq:
                x_best = x
                best_acq = val

        # Return the input that maximizes the acquisition function.
        return x_best

    @abstractmethod
    def evaluate(self, X):
        """Evaluate the acquisition function at the specified inputs. Unlike the
        gradient computation for the acquisition function, this method supports
        matrix-like inputs representing multiple locations at which to evaluate
        the acquisition function.

        Parameters:
            X (numpy array): A two-dimensional numpy array that represents the
                row-wise inputs in the unit hypercube that should be used as
                input to the acquisition function.

        Returns:
            The value of the acquisition function at the specified inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def grad_input(self, x):
        """Compute the gradient of the acquisition function with respect to the
        inputs. This method returns the gradient of the acquisition, not the
        gradient of the negative of the acquisition function.

        Parameters:
            x (numpy array): An input in the unit hypercube at which the
                gradient should be computed.

        Returns:
            The gradient of the acquisition function at the specified input.
        """
        raise NotImplementedError()
