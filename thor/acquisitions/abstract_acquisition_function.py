from abc import ABCMeta, abstractmethod


class AbstractAcquisitionFunction(object):
    """Abstract Acquisition Function Class"""
    __metaclass__ = ABCMeta
    def __init__(self, model):
        """Initialize parameters of the abstract acquisition function object."""
        self.model = model

    @abstractmethod
    def acquire(self, X_pred):
        """Compute the acquisition using the estimated model at the locations in
        a matrix of query locations.
        """
        raise NotImplementedError()

    @abstractmethod
    def grad_input(self, x):
        """Compute the gradient of the acquisition function with respect to the
        inputs.
        """
        raise NotImplementedError()
