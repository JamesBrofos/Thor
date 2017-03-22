import numpy as np
from scipy.stats import norm
from .improvement_acquisition_function import ImprovementAcquisitionFunction


class ExpectedImprovement(ImprovementAcquisitionFunction):
    """Expected Improvement Acquisition Function Class"""
    def acquire(self, X_pred):
        """Implementation of abstract base class method."""
        gamma, mean, var = self.score(X_pred)
        sd = np.sqrt(var)
        ei = (mean - self.y_best) * norm.cdf(gamma) + sd * norm.pdf(gamma)
        return ei

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        gamma, mean, var = self.score(x)
        sd = np.sqrt(var)
        d_mean, d_sd = self.model.grad_input(x)
        d_gamma = (d_mean - gamma * d_sd) / sd
        grad = (gamma * norm.cdf(gamma) + norm.pdf(gamma)) * d_sd
        grad += sd * norm.cdf(gamma) * d_gamma

        return grad
