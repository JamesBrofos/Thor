import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def negative_acquisition_function(params, acquisition):
    params = np.atleast_2d(params)
    return -acquisition.acquire(params)

def negative_acquisition_function_grad(params, acquisition):
    params = np.atleast_2d(params)
    return -acquisition.grad_input(params)


class BayesianOptimization(object):
    """Bayesian Optimization Class"""
    def __init__(self, acquisition, space):
        """Initialize parameters of the Bayesian optimization object."""
        self.acquisition = acquisition
        self.space = space

    def __maximize_acquisition(self):
        """Helper function that leverages the BFGS algorithm with a bounded
        input space in order to converge the maximum of the acquisition function
        using gradient ascent. Notice that this function returns the point in
        the original input space, not the point in the unit hypercube.
        """
        k = self.acquisition.model.X.shape[1]
        bounds = [(0., 1.)] * k
        x = np.random.uniform(low=0., high=1., size=(1, k))
        res = fmin_l_bfgs_b(
            negative_acquisition_function,
            x,
            fprime=negative_acquisition_function_grad,
            args=(self.acquisition, ),
            bounds=bounds,
            disp=0
        )
        x_max = np.atleast_2d(res[0])

        return self.space.invert(x_max.ravel()), self.acquisition.acquire(x_max)

    def recommend(self, n_evals):
        """Choose points to evaluate from the parameter space based on Bayesian
        optimization. This function uses multiple random restarts in the unit
        hypercube in order to identify local maxima of the acquisition function.
        """
        # TODO: Multiple random restarts.
        k = self.acquisition.model.X.shape[1]
        recommendations = np.zeros((n_evals, k))
        acquisition = np.zeros((n_evals, ))
        for i in range(n_evals):
            recommendations[i], acquisition[i] = self.__maximize_acquisition()

        return recommendations, acquisition
