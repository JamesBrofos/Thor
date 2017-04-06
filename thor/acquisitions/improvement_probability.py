import numpy as np
from scipy.stats import norm
from .improvement_acquisition_function import ImprovementAcquisitionFunction


class ImprovementProbability(ImprovementAcquisitionFunction):
    """Improvement Probability Acquisition Function Class"""
    def __init__(self, model, db_acq, y_best=None):
        """Initialize parameters of the improvement probability acquisition
        function.
        """
        super(ImprovementProbability, self).__init__(
            model, db_acq, y_best=y_best
        )

    def evaluate(self, X):
        """Implementation of abstract base class method."""
        return norm.cdf(self.score(X)[0])

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        gamma, mean, sd = self.score(x)
        d_mean, d_sd = self.model.grad_input(x)
        d_gamma = (d_mean - gamma * d_sd) / sd

        return norm.pdf(gamma) * d_gamma


