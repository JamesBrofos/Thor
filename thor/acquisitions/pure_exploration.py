import numpy as np
from .abstract_gradient_acquisition_function import (
    AbstractGradientAcquisitionFunction
)


class PureExploration(AbstractGradientAcquisitionFunction):
    """Pure Exploration Acquisition Function"""
    def evaluate(self, X):
        """Implementation of abstract base class method."""
        return self.model.predict(X)[1]

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        return self.model.grad_input(x)[1]
