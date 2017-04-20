import numpy as np
from multiprocessing import Pool
from scipy.optimize import fmin_l_bfgs_b
from ..gaussian_process import GaussianProcess
from ...kernels.squared_exponential_kernel import SquaredExponentialKernel


def __negative_marginal_likelihood(params, X, y, kernel):
    # Create Gaussian process.
    kernel.update(params)
    gp = GaussianProcess(kernel)
    gp.fit(X, y)

    return -1. * gp.log_likelihood()

def __negative_marginal_likelihood_grad(params, X, y, kernel):
    # Create Gaussian process.
    kernel.update(params)
    gp = GaussianProcess(kernel)
    gp.fit(X, y)
    # Compute the gradient and make sure that the gradient with respect to the
    # length scales is treated as an array.
    grad_dict = gp.grad_log_likelihood()
    grad = -np.concatenate([grad_dict[p.name] for p in kernel.parameters])

    return grad



def fit_marginal_likelihood(
        X, y, n_restarts, kernel
):
        """Fit the parameters of the Gaussian process by maximizing the marginal
        log-likelihood of the data.
        """
        # Initialize best log-likelihood observed so far.
        best_ll = -np.inf
        # Initialize search bounds.
        bounds = kernel.bounds

        for i in range(n_restarts):
            # Keep track of progress.
            print("Progress:\t{} / {}".format(i, n_restarts))
            # Randomly generate kernel parameters.
            params = kernel.sample()
            init_params = np.concatenate(
                [params[p.name] for p in kernel.parameters]
            )
            # Train the model.
            try:
                # Minimize the negative marginal likelihood.
                res = fmin_l_bfgs_b(
                    __negative_marginal_likelihood,
                    init_params,
                    fprime=__negative_marginal_likelihood_grad,
                    args=(X, y, kernel),
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
                gp = GaussianProcess(kernel)
                gp.fit(X, y)

                # Keep track of the kernel parameters corresponding to the best
                # model learned so far.
                cur_ll = gp.log_likelihood()
                if cur_ll > best_ll:
                    best_ll = cur_ll
                    best_params = bfgs_params
                    print("Best log-likelihood:\t{}".format(best_ll))

        # Set the parameters of the Gaussian process according to the kernel
        # parameters.
        kernel.update(best_params)
        gp = GaussianProcess(kernel)
        gp.fit(X, y)

        # Return the Gaussian process whose kernel parameters were estimated via
        # maximum marginal likelihood.
        return gp

