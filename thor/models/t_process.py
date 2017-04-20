import numpy as np
import scipy.linalg as spla

# TODO: In the rewrite of this part of the library, notice that there is a lot
#       of repeated code from the Gaussian process implementation.

class StudentProcess(object):
    """Student-t Process Class"""
    def __init__(self, kernel):
        """Initialize the parameters of the Student-t process object."""
        self.kernel = kernel

    def fit(self, X, y):
        """Estimate the parameters of the Student-t process based on available
        training data.
        """
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y - self.prior_mean
        # Compute the covariance matrix of the observed inputs.
        K = self.kernel.cov(self.X)
        # For a numerically stable algorithm, we use Cholesky decomposition.
        self.__L = spla.cholesky(K, lower=True)
        self.__alpha = spla.cho_solve((self.__L, True), self.y).ravel()
        L_inv = spla.solve_triangular(self.__L.T, np.eye(self.__L.shape[0]))
        self.__K_inv = L_inv.dot(L_inv.T)
        self.beta = y.dot(self.__K_inv).dot(y)



