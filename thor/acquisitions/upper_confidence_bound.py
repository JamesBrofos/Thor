import numpy as np
from .abstract_acquisition_function import AbstractAcquisitionFunction


class UpperConfidenceBound(AbstractAcquisitionFunction):
    """Upper Confidence Bound Acquisition Function Class"""
    def __init__(self, model, kappa=2.):
        """Initialize parameters of the upper confidence bound acquisition
        function object.
        """
        super(UpperConfidenceBound, self).__init__(model)
        self.kappa = kappa

    def acquire(self, X_pred):
        """Implementation of abstract base class method."""
        mean, sd = self.model.predict(X_pred)
        return mean + self.kappa * sd

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        d_mean, d_sd = self.model.grad_input(x)
        return d_mean + self.kappa * d_sd
