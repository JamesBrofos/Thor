import numpy as np
from scipy.stats import norm
from .improvement_acquisition_function import ImprovementAcquisitionFunction


class ImprovementProbability(ImprovementAcquisitionFunction):
    """Improvement Probability Acquisition Function Class"""
    def acquire(self, X_pred):
        """Implementation of abstract base class method."""
        return norm.cdf(self.score(X_pred)[0])

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        gamma, mean, sd = self.score(x)
        d_mean, d_sd = self.model.grad_input(x)
        d_gamma = (d_mean - gamma * d_sd) / sd

        return norm.pdf(gamma) * d_gamma


