import numpy as np
from .abstract_gradient_acquisition_function import (
    AbstractGradientAcquisitionFunction
)


class UpperConfidenceBound(AbstractGradientAcquisitionFunction):
    """Upper Confidence Bound Acquisition Function Class"""
    def __init__(self, model, db_acq, kappa=2.):
        """Initialize parameters of the upper confidence bound acquisition
        function object.
        """
        super(UpperConfidenceBound, self).__init__(
            model, db_acq
        )
        self.kappa = kappa

    def evaluate(self, X):
        """Implementation of abstract base class method."""
        mean, sd = self.model.predict(X)
        return mean + self.kappa * sd

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        d_mean, d_sd = self.model.grad_input(x)
        return d_mean + self.kappa * d_sd
