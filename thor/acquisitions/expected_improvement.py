import numpy as np
from scipy.stats import norm
from .improvement_acquisition_function import ImprovementAcquisitionFunction


class ExpectedImprovement(ImprovementAcquisitionFunction):
    """Expected Improvement Acquisition Function Class"""
    def __init__(self, model, db_acq, y_best=None):
        """Initialize parameters of the expected improvement acquisition
        function.
        """
        super(ExpectedImprovement, self).__init__(
            model, db_acq, y_best=y_best
        )

    def evaluate(self, X):
        """Implementation of abstract base class method."""
        gamma, mean, sd = self.score(X)
        ei = (mean - self.y_best) * norm.cdf(gamma) + sd * norm.pdf(gamma)
        return ei

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        gamma, mean, sd = self.score(x)
        d_mean, d_sd = self.model.grad_input(x)
        d_gamma = (d_mean - gamma * d_sd) / sd
        grad = (gamma * norm.cdf(gamma) + norm.pdf(gamma)) * d_sd
        grad += sd * norm.cdf(gamma) * d_gamma

        return grad
