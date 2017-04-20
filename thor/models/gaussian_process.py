import numpy as np
import scipy.linalg as spla
from ..kernels import SumKernel


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
        self.__alpha = spla.cho_solve((self.__L, True), self.y).ravel()
        L_inv = spla.solve_triangular(self.__L.T, np.eye(self.__L.shape[0]))
        self.__K_inv = L_inv.dot(L_inv.T)

    def predict(self, X_pred, covariance=False):
        """Leverage Bayesian posterior inference to compute the predicted mean
        and variance of a given set of inputs given the available training data.
        Notice that it is necessary to first fit the Gaussian process model
        before posterior inference can be performed.
        """
        # Compute the cross covariance between training and the requested
        # inference locations. Also compute the covariance matrix of the
        # observed inputs and the covariance at the inference locations.
        if type(self.kernel) == SumKernel:
            K_pred = self.kernel.cov(X_pred, include_noise=False)
        else:
            K_pred = self.kernel.cov(X_pred)
        K_cross = self.kernel.cov(X_pred, self.X)
        v = spla.solve_triangular(self.__L, K_cross.T, lower=True)
        # Posterior inference. Notice that we add a small amount of noise to the
        # diagonal for regulatization purposes.
        mean = K_cross.dot(self.__alpha)
        cov = K_pred - v.T.dot(v) + 1e-8 * np.eye(K_pred.shape[0])
        # Compute the diagonal of the covariance matrix if we wish to disregard
        # all of the covariance information and only focus on the variances at
        # the given inputs.
        if covariance:
            return mean, cov
        else:
            return mean, np.sqrt(np.diag(cov))

    def sample(self, X):
        """Sample target variables from the predictive posterior distribution of
        the Gaussian process.
        """
        # Get the mean and covariance.
        mean, cov = self.predict(X, covariance=True)
        # If there is noise in the kernel, make sure we are sampling from the
        # predictive distribution of the targets and not the mean.
        if type(self.kernel) == SumKernel:
            cov += self.kernel.noise_kernel.cov(X)

        return np.random.multivariate_normal(mean, cov)

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
        grad_params = self.kernel.grad_params(self.X)
        # Helpful matrices (avoids need to recompute).
        A = np.outer(self.__alpha, self.__alpha)
        D = A - self.__K_inv
        n = D.shape[0]
        # Define log-likelihood gradient dictionary.
        ll_grad = {}
        # Compute and return gradients.
        for p, grad in grad_params.items():
            ll_grad[p] = np.atleast_1d(0.5 * np.trace(D.dot(grad))) / n

        return ll_grad

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
        sd = self.predict(x)[1]
        K_cross = self.kernel.cov(x, self.X)
        M = spla.cho_solve((self.__L, True), K_cross.T).ravel()
        d_sd = -d_kernel.T.dot(M) / sd

        return d_mean, d_sd

