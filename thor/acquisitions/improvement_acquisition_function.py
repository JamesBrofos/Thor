import numpy as np
from .abstract_acquisition_function import AbstractAcquisitionFunction


class ImprovementAcquisitionFunction(AbstractAcquisitionFunction):
    """Improvement-Based Acquisition Function Class"""
    def __init__(self, model, y_best=None):
        """Initialize parameters of the improvement-based acquisition function.
        """
        super(ImprovementAcquisitionFunction, self).__init__(model)
        self.y_best = y_best if y_best is not None else self.model.y.max()

    def score(self, X_pred):
        """Compute a z-score quantity for the prediction at a given input. This
        allows us to balance improvement over the current best while controlling
        for uncertainty.
        """
        # Compute the mean and standard deviation of the model's interpolant of
        # the objective function.
        mean, sd = self.model.predict(X_pred)
        # Compute z-score-like quantity capturing the excess of the mean over
        # the current best, adjusted for uncertainty in the measurement.
        gamma = (mean - self.y_best) / sd

        return gamma, mean, sd
