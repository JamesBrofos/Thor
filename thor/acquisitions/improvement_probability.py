import numpy as np
from scipy.stats import norm
from .improvement_acquisition_function import ImprovementAcquisitionFunction


class ImprovementProbability(ImprovementAcquisitionFunction):
    """Improvement Probability Acquisition Function Class

    The improvement probability acquisition function leverages the idea of
    probability of improvement to select the next hyperparameter configuration
    to evaluate. This acquisition function accumulates the probability mass
    above the current best observation given the posterior distribution computed
    by the surrogate model. Because it does not weight the extent of the
    improvement (only its probability), the probability of improvement can be
    prone to exploiting too much in practical applications.
    """
    def evaluate(self, X):
        """Implementation of abstract base class method."""
        return norm.cdf(self.score(X)[0])

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        gamma, mean, sd = self.score(x)
        d_mean, d_sd = self.model.grad_input(x)
        d_gamma = (d_mean - gamma * d_sd) / sd

        return norm.pdf(gamma) * d_gamma


