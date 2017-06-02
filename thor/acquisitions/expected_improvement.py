import numpy as np
from scipy.stats import norm
from .improvement_acquisition_function import ImprovementAcquisitionFunction


class ExpectedImprovement(ImprovementAcquisitionFunction):
    """Expected Improvement Acquisition Function Class

    The expected improvement acquisition function, like the probability of
    improvement, similarly relies on the concept of improvement: That is, the
    extent to which the metric of interest can be expected under the posterior
    to exceed the current observed maximum. The expected improvement does not
    merely accumulate the probability density above the best value of the
    metric, however; instead, it weights each hyperparameter configuration
    according to the extent of the improvement. Therefore, improvement must not
    only be likely, but it must also be substantive.

    In practice, the expected improvement has been shown to out perform the
    upper confidence bound and probability of improvement acquisition functions.
    It is less likely to exploit local maxima than the probability of
    improvement is, and does not rely on its own hyperparameters unlike the
    upper confidence bound.
    """
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
