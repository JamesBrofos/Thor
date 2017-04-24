import numpy as np
import json
from .abstract_acquisition_function import AbstractAcquisitionFunction
from .improvement_probability import ImprovementProbability
from .expected_improvement import ExpectedImprovement
from .upper_confidence_bound import UpperConfidenceBound
from .pure_exploration import PureExploration


class HedgeAcquisition(AbstractAcquisitionFunction):
    """Hedge Acquisition Function Class"""
    def __init__(self, model, db_acq, constituents=[
            ImprovementProbability,
            ExpectedImprovement,
            UpperConfidenceBound,
    ]):
        """Initialize parameters of the hedge acquisition function object."""
        super(HedgeAcquisition, self).__init__(model, db_acq)
        self.constituents = [acq(self.model, None) for acq in constituents]
        self.n_constituents = len(self.constituents)
        # Create an internal variable to recall which points were selected for
        # candidates at the previous iteration by each of the constituent
        # acquisition functions.
        try:
            D = json.loads(self.db_acq.params)
            self.prev_X_select = np.array(D["prev_X"])
            self.weights = np.array(D["weights"])
            self.n_iters = D["n_iters"]
        except TypeError:
            self.prev_X_select = None
            self.weights = np.zeros((self.n_constituents, ))
            self.n_iters = 1

    def select(self):
        """Implementation of abstract base class method."""
        # Compute the candidate recommendations from each constituent
        # acquisition function.
        X_select = np.atleast_2d([acq.select() for acq in self.constituents])
        if self.prev_X_select is not None:
            # Update the weights using the previously selected candidates but
            # before selecting the latest configuration to recommend.
            updates = self.model.predict(self.prev_X_select)[0]
            self.weights += updates
            print(updates)
            print(self.__weights_to_probs())

        # Choose one of the candidates to evaluate.
        p = self.__weights_to_probs()
        x_chosen = X_select[np.random.choice(self.n_constituents, p=p)]
        # Note that the location of this assignment matters. At the first
        # iteration, there will be no previous selection, but at the conclusion
        # of the call to maximize, there will be.
        self.db_acq.params = json.dumps({
            "prev_X": X_select.tolist(),
            "weights": self.weights.tolist(),
            "n_iters": self.n_iters + 1
        })

        return x_chosen

    def __weights_to_probs(self):
        """Convert the weights to a vector of probabilities in order to randomly
        select an acquisition function whose recommendation is chosen.
        Probabilities are computed via a scaled softmax.
        """
        nu = np.sqrt(8. * np.log(self.n_constituents) / self.n_iters)
        w = nu * self.weights
        e = np.exp(w - w.max())
        return e / e.sum()
