from .abstract_acquisition_function import AbstractAcquisitionFunction
from .improvement_probability import ImprovementProbability
from .expected_improvement import ExpectedImprovement
from .upper_confidence_bound import UpperConfidenceBound


class HedgeAcquisition(AbstractAcquisitionFunction):
    """Hedge Acquisition Function Class"""
    def __init__(self, model, constituents=[
            ImprovementProbability, ExpectedImprovement, UpperConfidenceBound
    ], nu=1.0):
        """Initialize parameters of the hedge acquisition function object."""
        super(HedgeAcquitision, self).__init__(model)
        self.constituents = [acq(self.model) for acq in constituents]
        self.n_constituents = len(self.constituents)
        self.weights = np.zeros((self.n_constituents, ))
        self.nu = nu

    def select(self, x):
        """Implementation of abstract base class method."""
        # Compute the candidate recommendations from each constituent
        # acquisition function.
        x_select = np.atleast_2d([acq.select(x) for acq in self.constituents])
        # Choose one of the candidates to evaluate.
        p = self.__weights_to_probs()
        x_chosen = X_select[np.random.choice(self.n_constituents, p=p)]
        # Update the weights.
        self.weights += self.model.predict(x_select)[0]

    def __weights_to_probs(self):
        """Convert the weights to a vector of probabilities in order to randomly
        select an acquisition function whose recommendation is chosen.
        """
        e = np.exp(self.nu * self.weights)
        return e / e.sum()
