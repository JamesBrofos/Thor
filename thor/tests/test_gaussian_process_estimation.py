import unittest
import numpy as np
import matplotlib.pyplot as plt
from thor.kernels import (
    SquaredExponentialKernel,
    MaternKernel,
    NoiseKernel,
    SumKernel
)
from thor.models import GaussianProcess
from thor.models.tuning import fit_marginal_likelihood


class GaussianProcessEstimationTest(unittest.TestCase):
    """Test Module for Gaussian Process Estimation"""
    def test_gaussian_process_estimation(self):
        # Create fake data.
        n = 200
        k = 1
        X = np.random.uniform(size=(n, k))
        # Create kernel function.
        true_kernel = MaternKernel(1., (0.1, ))
        true_mean, true_cov = np.zeros((n, )), true_kernel.cov(X)
        y = np.random.multivariate_normal(true_mean, true_cov)

        # Create a kernel whose parameters are initialized to null values.
        dom_kernel = MaternKernel(np.nan, np.full((k, ), np.nan))
        noise_kernel = NoiseKernel(np.nan)
        kernel = SumKernel([dom_kernel], noise_kernel)
        # Fit the Gaussian process with marginal likelihood.
        n_restarts = 5
        prior_mean = 0.
        gp = fit_marginal_likelihood(X, y, n_restarts, kernel, prior_mean)

        print(gp.kernel.kernels[0].length_scales.value)
        print(gp.kernel.kernels[0].amplitude.value)
        print(gp.kernel.noise_kernel.noise.value)

        # Predict with the Gaussian process.
        n_pred = 1000
        X_pred = np.atleast_2d(np.linspace(0., 1., num=n_pred)).T
        mean, sd = gp.predict(X_pred)

        # Visualize.
        if True:
            plt.figure(figsize=(18, 6))
            plt.plot(X.ravel(), y, "k.")
            plt.plot(X_pred.ravel(), mean, "r-")
            plt.plot(X_pred.ravel(), mean + 2*sd, "r--")
            plt.plot(X_pred.ravel(), mean - 2*sd, "r--")
            plt.grid()
            plt.show()


if __name__ == "__main__":
    unittest.main()
