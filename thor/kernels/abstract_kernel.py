from abc import ABCMeta, abstractmethod, abstractproperty
from functools import reduce


class AbstractKernel(object):
    """Abstract Kernel Class

    Kernel functions are leveraged by probabilistic processes for computing the
    covariance between inputs. Common examples of kernel functions include the
    Matern kernel and the squared exponential; kernels can also be computed as
    direct sums of other kernels, and noise can be introduced into the kernel
    in order to reflect stochastic outputs.

    This class implements several utilities for working with kernel functions
    for application to Bayesian optimization. In particular, the class supports
    sampling from the kernel parameters, computing the covariance between two
    matrix-valued inputs, as well as functions for computing the gradient of the
    kernel with respect to both kernel parameters and vector inputs.
    """
    __metaclass__ = ABCMeta

    def sample(self):
        """Sample kernel parameters. Sampling the kernel is done by sampling
        each hyperparameter individually.
        """
        return {p.name: p.sample() for p in self.parameters}

    def update(self, values):
        """Update the parameters of the kernel using provided values. This
        method is used during gradient ascent procedures to update the kernel.

        Parameters:
            values (numpy array): A numpy array containing the updated values
                for each of the parameters of the kernel.
        """
        # This index keeps track of where we are in the update vector as we
        # proceed to update the values of each kernel hyperparameter.
        idx = 0
        # Iterate over each hyperparameter of the kernel and upate its value.
        for p in self.parameters:
            sz = len(p.value)
            p.value = values[idx:(idx+sz)]
            idx += sz

    @property
    def bounds(self):
        """Retrieve the bounds of the kernel parameters. Note that the bounds of
        the kernel are the hyper-rectangle formed by concatenating the bounds of
        each parameter of the kernel. This is used when trying to maximize the
        log-likelihood function of a probabilistic process as a given of kernel
        hyperparameters.
        """
        bnds = []
        for p in self.parameters:
            for _ in range(len(p.value)):
                bnds.append(p.bounds)

        return bnds

    @abstractproperty
    def parameters(self):
        """Returns the parameters of the kernel, including such values as the
        length scales, the amplitude, and the noise.

        Raises:
            NotImplementedError: As an abstract method, all classes inheriting
                from the abstract kernel class must implement this property.
        """
        raise NotImplementedError()

    @abstractmethod
    def cov(self, X, Y=None):
        """Any kernel must implement a covariance (or, indeed, a cross
        covariance) function between inputs. Notice that this is independent of
        the outputs.

        Parameters:
            X (numpy array): A two-dimensional numpy array. This represents a
                set of inputs at which to compute the covariance of the target
                according to the probabilistic process.
            Y (numpy array, optional): If provided, this method will compute the
                cross-covariance between the rows in `X` and the rows in `Y`. If
                this argument is not provided, then only the covariance among
                the rows of `X` is computed.

        Examples:
            The following are equivalent.

            >>> kernel.cov(X)
            >>> kernel.cov(X, X)

        Raises:
            NotImplementedError: As an abstract method, all classes inheriting
                from the abstract kernel class must implement this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def grad_input(self, x, Y):
        """Compute the gradient of the kernel with respect to the first input of
        the covariance function and essentially conditioned on the set of
        observed inputs.

        Parameters:
            x (numpy array): A one-dimensional numpy array representing a
                location in the input space. This function computes the gradient
                of the covariance matrix with respect to infinitesimally small
                changes in this input.
            Y (numpy array): A two-dimensional numpy array representing the
                observed inputs.

        Returns:
            A the gradient of the covariance matrix with respect to changes in
                the input argument `x`.

        Raises:
            NotImplementedError: As an abstract method, all classes inheriting
                from the abstract kernel class must implement this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def grad_params(self, X):
        """Compute the gradient of the kernel with respect to the kernel
        parameters: the amplitude, the length scales, and the noise variance.

        This function will return gradients with respect the kernel parameters
        in the order above.

        Parameters:
            X (numpy array): A two-dimensional numpy array representing the
                observed inputs.

        Returns:
            A dictionary containing the gradients of the covariance matrix of
                the observed inputs with respect to each of the parameters of
                the kernel.

        Raises:
            NotImplementedError: As an abstract method, all classes inheriting
                from the abstract kernel class must implement this method.

        """
        raise NotImplementedError()
