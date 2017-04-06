from abc import ABCMeta, abstractmethod


class AbstractAcquisitionFunction(object):
    """Abstract Acquisition Function Class"""
    __metaclass__ = ABCMeta
    def __init__(self, model, db_acq):
        """Initialize parameters of the abstract acquisition function object."""
        self.model = model
        self.db_acq = db_acq

    @abstractmethod
    def maximize(self, n_evals):
        """Compute the acquisition using the estimated model at the locations in
        a matrix of query locations.
        """
        raise NotImplementedError()

