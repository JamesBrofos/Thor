import numpy as np
import scipy.linalg as spla
from ..kernels import SquaredExponentialKernel


class GaussianProcess(object):
    """Gaussian Process Class"""
    def __init__(self, kernel):
        """Initialize the parameters of the Gaussian process object."""
        self.kernel = kernel

    def fit(self, X, y):
        """Estimate the parameters of the Gaussian process based on available
        training data.
        """
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y
        # Compute the covariance matrix of the observed inputs.
        K = self.kernel.cov(self.X)
        # For a numerically stable algorithm, we use Cholesky decomposition.
        self.__L = spla.cholesky(K, lower=True)
        self.__alpha = spla.cho_solve((self.__L, True), self.y)
        L_inv = spla.solve_triangular(self.__L.T, np.eye(self.__L.shape[0]))
        self.__K_inv = L_inv.dot(L_inv.T)

    def predict(self, X_pred, diagonalize=True):
        """Leverage Bayesian posterior inference to compute the predicted mean
        and variance of a given set of inputs given the available training data.
        Notice that it is necessary to first fit the Gaussian process model
        before posterior inference can be performed.
        """
        # Compute the cross covariance between training and the requested
        # inference locations. Also compute the covariance matrix of the
        # observed inputs and the covariance at the inference locations.
        K_pred = self.kernel.cov(X_pred)
        K_cross = self.kernel.cov(X_pred, self.X)
        v = spla.solve_triangular(self.__L, K_cross.T, lower=True)
        # Posterior inference.
        mean = K_cross.dot(self.__alpha)
        var = K_pred - v.T.dot(v)
        # Compute the diagonal of the covariance matrix if we wish to disregard
        # all of the covariance information and only focus on the variances at
        # the given inputs.
        if diagonalize:
            var = np.diag(var)

        return mean, var

    def log_likelihood(self):
        """Compute the log-likelihood of the data under the Gaussian process
        model with the given length scales, amplitude, and noise level of the
        kernel.
        """
        return (
            -0.5 * self.y.dot(self.__alpha) -
            np.sum(np.log(np.diag(self.__L))) -
            0.5 * self.X.shape[0] * np.log(2.*np.pi)
        )

    def grad_log_likelihood(self):
        """Compute the gradient of the log-likelihood of the data under the
        Gaussian process model. This is used to estimate the kernel parameters
        such as the amplitude, the length scales, and the noise variance in a
        principled manner.
        """
        # Gradients of the kernel with respect to kernel parameters.
        K_amp_grad, K_ls_grad = self.kernel.grad_params(self.X)
        # Helpful matrices (avoids need to recompute).
        A = np.outer(self.__alpha, self.__alpha)
        D = A - self.__K_inv
        # Compute and return gradients.
        amp_grad = 0.5 * np.trace(D.dot(K_amp_grad))
        ls_grad = 0.5 * np.trace(D.dot(K_ls_grad))
        noise_grad = 0.5 * np.trace(D)

        return amp_grad, ls_grad, noise_grad

    def grad_input(self, x):
        """Compute the gradient of the mean function and the standard deviation
        function at the provided input.
        """
        # Compute the gradient of the mean function.
        d_kernel = self.kernel.grad_input(x, self.X)
        d_mean = d_kernel.T.dot(self.__alpha)
        # Compute the gradient of the standard deviation function. It is
        # absolutely crucial to note that the predict method returns the
        # variance, not the standard deviation, of the prediction.
        sd = np.sqrt(self.predict(x)[1])
        K_cross = self.kernel.cov(x, self.X)
        M = spla.cho_solve((self.__L, True), K_cross.T).ravel()
        d_sd = -d_kernel.T.dot(M) / sd

        return d_mean, d_sd

