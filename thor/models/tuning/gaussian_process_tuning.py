import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from ..gaussian_process import GaussianProcess
from ...kernels.squared_exponential_kernel import SquaredExponentialKernel


def __negative_marginal_likelihood(params, X, y, kernel_class, prior_mean):
    # Extract kernel parameters.
    amplitude = params[0]
    noise = params[-1]
    length_scales = params[1:-1]
    # Create Gaussian process kernel.
    kernel = kernel_class(amplitude, length_scales, noise)
    gp = GaussianProcess(kernel, prior_mean)
    gp.fit(X, y)

    return -1. * gp.log_likelihood()

def __negative_marginal_likelihood_grad(params, X, y, kernel_class, prior_mean):
    # Extract kernel parameters.
    amplitude = params[0]
    noise = params[-1]
    length_scales = params[1:-1]
    # Create Gaussian process kernel.
    kernel = kernel_class(amplitude, length_scales, noise)
    gp = GaussianProcess(kernel, prior_mean)
    gp.fit(X, y)
    # Compute the gradient and make sure that the gradient with respect to the
    # length scales is treated as an array.
    amp_grad, ls_grad, noise_grad = gp.grad_log_likelihood()
    grad = -np.insert(ls_grad, (0, -1), (amp_grad, noise_grad))

    return grad

def fit_marginal_likelihood(
        X, y, n_restarts, kernel_class, prior_mean
):
        """Fit the parameters of the Gaussian process by maximizing the marginal
        log-likelihood of the data.
        """
        best_ll = -np.inf
        for i in range(n_restarts):
            # Randomly generate an amplitude, length scales, and noise level.
            amplitude = np.random.uniform(0., 2.)
            length_scales = np.random.uniform(0., 1., size=(X.shape[1], ))
            noise = np.random.uniform(0., 0.5)
            # Train the model.
            try:
                # Create parameter vector for the BFGS algorithm
                params = np.insert(length_scales, (0, -1), (amplitude, noise))
                # Set search bounds.
                bounds = [(0., 2.)]
                for _ in range(X.shape[1]):
                    bounds.append((0.001, 1.))
                bounds.append((0.0, 0.5))
                # Minimize the negative marginal likelihood.
                res = fmin_l_bfgs_b(
                    __negative_marginal_likelihood,
                    params,
                    fprime=__negative_marginal_likelihood_grad,
                    args=(X, y, kernel_class, prior_mean),
                    bounds=bounds,
                    disp=0
                )
                amplitude, length_scales, noise = (
                    res[0][0], res[0][1:-1], res[0][-1]
                )
                kernel = kernel_class(amplitude, length_scales, noise)
                gp = GaussianProcess(kernel, prior_mean)
                gp.fit(X, y)
            except np.linalg.linalg.LinAlgError:
                print("Linear algebra failure.")
                continue
            else:
                # Keep track of the kernel parameters corresponding to the best
                # model learned so far.
                cur_ll = gp.log_likelihood()
                if cur_ll > best_ll:
                    best_ll = cur_ll
                    best_amplitude = amplitude
                    best_noise = noise
                    best_length_scales = length_scales
                    print("Best log-likelihood:\t{}".format(best_ll))

        # Set the parameters of the Gaussian process according to the kernel
        # parameters.
        kernel = kernel_class(
            best_amplitude, best_length_scales, best_noise
        )
        gp = GaussianProcess(kernel)
        gp.fit(X, y)

        # Return the Gaussian process whose kernel parameters were estimated via
        # maximum marginal likelihood.
        return gp

