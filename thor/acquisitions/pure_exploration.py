import numpy as np
from .abstract_gradient_acquisition_function import (
    AbstractGradientAcquisitionFunction
)


class PureExploration(AbstractGradientAcquisitionFunction):
    """Pure Exploration Acquisition Function

    Bayesian optimization balance exploration and exploitation. Exploration
    refers to the idea of looking in regions of high uncertainty for good
    hyperparameter configurations; exploitation refers to the idea of annealing
    in the area of local maxima to discover the best hyperparameter
    configurations. This acquisition function adopts a strategy of pure
    exploration; in particular, the acquisition function will select the point
    in the input space where it is maximally uncertain about the behavior of the
    underlying metric of interest.
    """
    def evaluate(self, X):
        """Implementation of abstract base class method."""
        return self.model.predict(X)[1]

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        return self.model.grad_input(x)[1]
