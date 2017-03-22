import unittest
import numpy as np
import matplotlib.pyplot as plt
from thor.models.tuning import fit_marginal_likelihood
from thor.kernels import SquaredExponentialKernel, MaternKernel


class GaussianProcessTuningTest(unittest.TestCase):
    def test_marginal_likelihood_tuning(self):
        np.random.seed(0)
        n = 50
        k = 2
        X = np.random.uniform(size=(n, k))
        ls = np.ones((k, )) * 0.1
        kernel_class = MaternKernel
        kernel = kernel_class(1., ls, 0.1)
        C = kernel.cov(X)
        y = np.random.multivariate_normal(np.zeros((n, )), C)
        if True:
            gp = fit_marginal_likelihood(
                X, y, kernel_class=kernel_class, n_restarts=30
            )

            if k == 1:
                n_pred = 100
                X_pred = np.atleast_2d(np.linspace(0., 1., num=n_pred)).T
                mean_pred, var_pred = gp.predict(X_pred)
                plt.figure()
                plt.plot(X, y, "k.")
                plt.plot(X_pred, mean_pred, "r-")
                plt.plot(X_pred, mean_pred + 2*np.sqrt(var_pred), "r--")
                plt.plot(X_pred, mean_pred - 2*np.sqrt(var_pred), "r--")
                plt.grid()
                plt.show()

if __name__ == "__main__":
    unittest.main()
