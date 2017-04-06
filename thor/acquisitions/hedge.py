import numpy as np
import json
from .abstract_acquisition_function import AbstractAcquisitionFunction
from .improvement_probability import ImprovementProbability
from .expected_improvement import ExpectedImprovement
from .upper_confidence_bound import UpperConfidenceBound


class HedgeAcquisition(AbstractAcquisitionFunction):
    """Hedge Acquisition Function Class"""
    def __init__(self, model, db_acq, constituents=[
            ImprovementProbability, ExpectedImprovement, UpperConfidenceBound
    ], nu=1.0):
        """Initialize parameters of the hedge acquisition function object."""
        super(HedgeAcquisition, self).__init__(model, db_acq)
        self.constituents = [acq(self.model, None) for acq in constituents]
        self.n_constituents = len(self.constituents)
        self.weights = np.zeros((self.n_constituents, ))
        self.nu = nu
        # Create an internal variable to recall which points were selected for
        # candidates at the previous iteration by each of the constituent
        # acquisition functions.
        try:
            self.prev_X_select = np.array(json.loads(self.db_acq.params)["prev_X"])
        except TypeError:
            self.prev_X_select = None

    def maximize(self, n_evals):
        """Implementation of abstract base class method."""
        # Compute the candidate recommendations from each constituent
        # acquisition function.
        X_select = np.atleast_2d(
            [acq.maximize(n_evals) for acq in self.constituents]
        )
        if self.prev_X_select is not None:
            # Update the weights using the previously selected candidates but
            # before selecting the latest configuration to recommend.
            self.weights += self.model.predict(self.prev_X_select)[0]
            print(self.__weights_to_probs())

        # Choose one of the candidates to evaluate.
        p = self.__weights_to_probs()
        x_chosen = X_select[np.random.choice(self.n_constituents, p=p)]
        # Note that the location of this assignment matters. At the first
        # iteration, there will be no previous selection, but at the conclusion
        # of the call to maximize, there will be.
        self.db_acq.params = json.dumps({"prev_X": X_select.tolist()})

        return x_chosen

    def __weights_to_probs(self):
        """Convert the weights to a vector of probabilities in order to randomly
        select an acquisition function whose recommendation is chosen.
        """
        e = np.exp(self.nu * self.weights)
        return e / e.sum()
