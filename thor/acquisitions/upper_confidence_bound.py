import numpy as np
from .abstract_gradient_acquisition_function import (
    AbstractGradientAcquisitionFunction
)


class UpperConfidenceBound(AbstractGradientAcquisitionFunction):
    """Upper Confidence Bound Acquisition Function Class

    The upper confidence bound algorithm uses the posterior credible interval as
    an acquisition function. In Thor, this posterior distribution is computed by
    either a Gaussian process interpolant or a post-hoc Bayesian linear
    regression layer on top of a neural network basis function regression. Under
    certain regularity conditions, the upper confidence bound can be shown to
    exhibit appealing properties in terms of the regret; this makes the upper
    confidence bound a useful algorithm for theoretical study. Unfortunately,
    the upper confidence bound also comes equipped with a hyperparameter that
    controls the balance between exploration and exploitation; changing this
    parameter will impact performance of the Bayesian optimization routine.

    Parameters:
        kappa (float, optional): A parameter controlling the acquisition
            functions preference for exploration as opposed to exploitation. A
            higher value will place more value on exploring regions of high
            uncertainty, whereas a value of zero is equivalent to pure
            exploitation. The default value of two corresponds to the
            acquisition function approximating the upper limit of the
            ninety-five percent credible interval under the probabilistic
            model's posterior.
    """
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
