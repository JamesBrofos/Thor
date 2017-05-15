from abc import ABCMeta, abstractmethod


class AbstractAcquisitionFunction(object):
    """Abstract Acquisition Function Class

    This class represents the most abstract form of the acquisition function
    utilized in Bayesian optimization. In particular, all acquisition functions
    must implement a select method which uses the information contained
    collected by the algorithm so far to intelligently rank hyperparameters in
    the domain according to their utility.

    Parameters:
        model (GaussianProcess): The probabilistic surrogate model that is used
            to interpolate the metric of interest as a function of the machine
            learning system's hyperparameters.
        db_acq (dict): A dictionary of information that is, presumably, stored
            on Thor Server's database. This includes relevant information about
            previous selections that may need to be retained and incorporated
            into the current selection.
    """
    __metaclass__ = ABCMeta
    def __init__(self, model, db_acq):
        """Initialize parameters of the abstract acquisition function object."""
        self.model = model
        self.db_acq = db_acq

    @abstractmethod
    def select(self):
        """Compute the acquisition using the estimated model at the locations in
        a matrix of query locations.

        Raises:
            NotImplementedError: As an abstract method, all classes inheriting
                from the abstract acquisition function class must implement this
                method.

        Returns:
            A numpy array representing a point in the unit hypercube that should
                be mapped into the original hyperparameter space and transmitted
                to the client for evaluation.
        """
        raise NotImplementedError()
