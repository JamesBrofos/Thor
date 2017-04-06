import numpy as np


class BayesianOptimization(object):
    """Bayesian Optimization Class"""
    def __init__(self, acquisition, space):
        """Initialize parameters of the Bayesian optimization object."""
        self.acquisition = acquisition
        self.space = space

    def recommend(self, n_evals):
        """Choose points to evaluate from the parameter space based on Bayesian
        optimization. This function uses multiple random restarts in the unit
        hypercube in order to identify local maxima of the acquisition function.
        """
        return self.acquisition.maximize(n_evals)
