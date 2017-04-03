import unittest
import matplotlib.pyplot as plt
import numpy as np
from thor.kernels import SquaredExponentialKernel, NoiseKernel, SumKernel
from thor.models import GaussianProcess
from thor.models.tuning import fit_marginal_likelihood


# For reproducibility.
np.random.seed(0)

class GaussianProcessTest(unittest.TestCase):
    """Test Module for Gaussian Process Class"""
    def test_multidimensional_gaussian_process(self):
        # Create fake data.
        n = 50
        k = 2
        X = np.random.uniform(size=(n, k))
        # Create kernel function.
        true_kernel = SquaredExponentialKernel(1., (0.1, 0.25))
        true_mean, true_cov = np.zeros((n, )), true_kernel.cov(X)
        y = np.random.multivariate_normal(true_mean, true_cov)

        # Create a kernel whose parameters are initialized to null values.
        dom_kernel = SquaredExponentialKernel(np.nan, np.full((k, ), np.nan))
        noise_kernel = NoiseKernel(np.nan)
        kernel = SumKernel([dom_kernel], noise_kernel)
        # Fit the Gaussian process with marginal likelihood.
        n_restarts = 500
        prior_mean = 0.
        gp = fit_marginal_likelihood(X, y, n_restarts, kernel, prior_mean)

    def test_fit_marginal_likelihood(self):
        # Create fake data.
        n = 5
        k = 1
        X = np.random.uniform(size=(n, k))
        y = np.sin(10*X).ravel() + np.random.normal(size=(n, )) / 10.
        # Create a kernel whose parameters are initialized to null values.
        dom_kernel = SquaredExponentialKernel(np.nan, np.full((k, ), np.nan))
        noise_kernel = NoiseKernel(np.nan)
        kernel = SumKernel([dom_kernel], noise_kernel)
        kernel = SquaredExponentialKernel(np.nan, np.full((k, ), np.nan))
        # Fit the Gaussian process with marginal likelihood.
        n_restarts = 100
        prior_mean = 0.
        gp = fit_marginal_likelihood(X, y, n_restarts, kernel, prior_mean)

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

    def test_squared_exponential_gaussian_process(self):
        # Define amplitude and length scale.
        amp, ls, noise = 1., (0.1, ), 0.01
        # Create kernel.
        dom_kernel = SquaredExponentialKernel(amp, ls)
        noise_kernel = NoiseKernel(noise)
        kernel = SumKernel([dom_kernel], noise_kernel)
        # Compute the Gaussian process.
        gp = GaussianProcess(kernel)
        # Create fake data.
        n = 50
        X = np.random.uniform(size=(n, 1))
        y = np.sin(10*X)
        # Train Gaussian process.
        gp.fit(X, y)

        # Predict with the Gaussian process.
        n_pred = 1000
        X_pred = np.atleast_2d(np.linspace(0., 1., num=n_pred)).T
        mean, sd = gp.predict(X_pred)

        # Visualize.
        if False:
            plt.figure(figsize=(18, 6))
            plt.plot(X.ravel(), y, "k.")
            plt.plot(X_pred.ravel(), mean, "r-")
            plt.plot(X_pred.ravel(), mean + 2*sd, "r--")
            plt.plot(X_pred.ravel(), mean - 2*sd, "r--")
            plt.grid()
            plt.show()


if __name__ == "__main__":
    unittest.main()


