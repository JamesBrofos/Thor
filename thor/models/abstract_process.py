import numpy as np
import scipy.linalg as spla
from scipy.optimize import fmin_l_bfgs_b
from abc import ABCMeta, abstractmethod, abstractproperty
from ..kernels import SumKernel


class AbstractProcess(object):
    """Abstract Probabilistic Process Class"""
    metaclass = ABCMeta

    def __init__(self, kernel):
        """Initialize the parameters of the abstract probabilistic process
        object.
        """
        self.kernel = kernel

    def fit(self, X, y):
        """Fit the parameters of the probabilistic process based on the
        available training data.
        """
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y
        # Compute the covariance matrix of the observed inputs.
        K = self.kernel.cov(self.X)
        # For a numerically stable algorithm, we use Cholesky decomposition.
        self.L = spla.cholesky(K, lower=True)
        self.alpha = spla.cho_solve((self.L, True), self.y).ravel()
        L_inv = spla.solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        self.K_inv = L_inv.dot(L_inv.T)
        self.beta = self.y.dot(self.alpha)

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
        v = spla.solve_triangular(self.L, K_cross.T, lower=True)
        # Posterior inference. Notice that we add a small amount of noise to the
        # diagonal for regulatization purposes.
        mean = K_cross.dot(self.alpha)
        cov = self.predict_prefactor * (
            K_pred - v.T.dot(v) + 1e-8 * np.eye(K_pred.shape[0])
        )
        # Compute the diagonal of the covariance matrix if we wish to disregard
        # all of the covariance information and only focus on the variances at
        # the given inputs.
        if covariance:
            return mean, cov
        else:
            return mean, np.sqrt(np.diag(cov))

    def sample(self, X):
        """Sample target variables from the predictive posterior distribution of
        the probabilistic process. Unlike the functions that will extend this
        function, this one will return the mean and variance of the posterior
        distribution over targets. It is up to the extensions to use these
        parameters as appropriate for sampling.
        """
        # Get the mean and covariance.
        mean, cov = self.predict(X, covariance=True)
        # If there is noise in the kernel, make sure we are sampling from the
        # predictive distribution of the targets and not the mean.
        if type(self.kernel) == SumKernel:
            cov += self.kernel.noise_kernel.cov(X)

        return mean, cov

    def grad_log_likelihood(self):
        """Compute the gradient of the log-likelihood of the data under the
        probabilistic process model. This is used to estimate the kernel
        parameters such as the amplitude, the length scales, and the noise
        variance in a principled manner.
        """
        # Gradients of the kernel with respect to kernel parameters.
        grad_params = self.kernel.grad_params(self.X)
        # Helpful matrices (avoids need to recompute).
        A = self.grad_prefactor * np.outer(self.alpha, self.alpha)
        D = A - self.K_inv
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
        d_mean = d_kernel.T.dot(self.alpha)
        # Compute the gradient of the standard deviation function. It is
        # absolutely crucial to note that the predict method returns the
        # variance, not the standard deviation, of the prediction.
        sd = self.predict(x)[1]
        K_cross = self.kernel.cov(x, self.X)
        M = spla.cho_solve((self.L, True), K_cross.T).ravel()
        d_sd = -d_kernel.T.dot(M) / sd

        return d_mean, d_sd

    @abstractmethod
    def log_likelihood(self):
        """Compute the log-likelihood of the data under the probabilistic
        process model with the given length scales, amplitude, and noise level
         of the kernel.
        """
        raise NotImplementedError()

    @abstractproperty
    def parameters(self):
        """The properties of the probabilistic process object."""
        raise NotImplementedError()

    @abstractproperty
    def predict_prefactor(self):
        """A multiplicative factor appearing in the predictive posterior
        calculation of probabilistic processes.
        """
        raise NotImplementedError()

    @abstractproperty
    def grad_prefactor(self):
        """A multiplicative factor appearing in the gradient calculation of
        probabilistic processes.
        """
        raise NotImplementedError()




def __negative_marginal_likelihood(params, X, y, kernel, model_class):
    # Create Gaussian process.
    kernel.update(params)
    model = model_class(kernel)
    model.fit(X, y)

    return -1. * model.log_likelihood()

def __negative_marginal_likelihood_grad(params, X, y, kernel, model_class):
    # Create Gaussian process.
    kernel.update(params)
    model = model_class(kernel)
    model.fit(X, y)
    # Compute the gradient and make sure that the gradient with respect to the
    # length scales is treated as an array.
    grad_dict = model.grad_log_likelihood()
    grad = np.concatenate([grad_dict[p.name] for p in kernel.parameters])

    return -1. * grad


def fit_marginal_likelihood(X, y, n_restarts, kernel, model_class):
        """Fit the parameters of the Gaussian process by maximizing the marginal
        log-likelihood of the data.
        """
        # Initialize best log-likelihood observed so far and initialize the
        # search bounds of the BFGS algorithm.
        best_ll = -np.inf
        bounds = kernel.bounds

        for i in range(n_restarts):
            # Keep track of progress.
            print("Progress:\t{} / {}".format(i, n_restarts))
            # Randomly generate kernel parameters.
            initial_params = np.concatenate(
                [p.sample() for p in kernel.parameters]
            )
            # Train the model.
            try:
                # Minimize the negative marginal likelihood.
                res = fmin_l_bfgs_b(
                    __negative_marginal_likelihood,
                    initial_params,
                    fprime=__negative_marginal_likelihood_grad,
                    args=(X, y, kernel, model_class),
                    bounds=bounds,
                    disp=0
                )
                bfgs_params = res[0]
            except np.linalg.linalg.LinAlgError:
                print("Linear algebra failure.")
                continue
            except UnboundLocalError:
                print("Unbound local variable.")
                continue
            else:
                # Update kernel parameters.
                kernel.update(bfgs_params)
                model = model_class(kernel)
                model.fit(X, y)

                # Keep track of the kernel parameters corresponding to the best
                # model learned so far.
                cur_ll = model.log_likelihood()
                if cur_ll > best_ll:
                    best_ll = cur_ll
                    best_params = bfgs_params
                    best_model = model
                    print("Best log-likelihood:\t{}".format(best_ll))

        # Set the parameters of the Gaussian process according to the kernel
        # parameters.
        kernel.update(best_params)
        model = model_class(kernel)
        model.fit(X, y)

        # Return the Gaussian process whose kernel parameters were estimated via
        # maximum marginal likelihood.
        return model



