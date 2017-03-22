import unittest
import numpy as np
import matplotlib.pyplot as plt
from thor.models import GaussianProcess
from thor.kernels import SquaredExponentialKernel, MaternKernel



class GaussianProcessTest(unittest.TestCase):
    def test_fit(self):
        n = 10
        k = 1
        X = np.random.uniform(low=-1., high=1., size=(n, k))
        ls = np.ones((k, )) * 0.4
        k = SquaredExponentialKernel(1., ls, 0.)
        C = k.cov(X)
        y = np.random.multivariate_normal(np.zeros((n, )), C)
        gp = GaussianProcess(k)
        gp.fit(X, y)
        n_pred = 100
        X_pred = np.atleast_2d(np.linspace(-1., 1., num=n_pred)).T
        mean_pred, var_pred = gp.predict(X_pred)

        self.assertEqual(n_pred, mean_pred.shape[0])
        self.assertEqual(n_pred, var_pred.shape[0])

        if False:
            plt.figure()
            plt.plot(X, y, "k.")
            plt.plot(X_pred, mean_pred, "r-")
            plt.plot(X_pred, mean_pred + 2*np.sqrt(var_pred), "r--")
            plt.plot(X_pred, mean_pred - 2*np.sqrt(var_pred), "r--")
            plt.grid()
            plt.show()

    def test_squared_exponential_grad_input(self):
        np.random.seed(0)
        n = 10
        k = 1
        X = np.random.uniform(low=-1., high=1., size=(n, k))
        ls = np.ones((k, )) * 0.1
        kernel = SquaredExponentialKernel(1., ls, 0.)
        C = kernel.cov(X)
        y = np.random.multivariate_normal(np.zeros((n, )), C)
        gp = GaussianProcess(kernel)
        gp.fit(X, y)
        n_pred = 100
        X_pred = np.atleast_2d(np.linspace(-1., 1., num=n_pred)).T
        mean_pred, var_pred = gp.predict(X_pred)

        x_mean_max = np.zeros((1, k))
        x_sd_max = np.zeros((1, k)) + 0.5
        for i in range(50):
            x_mean_max += 0.01 * gp.grad_input(x_mean_max)[0]
            x_sd_max += 0.01 * gp.grad_input(x_sd_max)[1]
        y_mean_max = gp.predict(x_mean_max)[0]
        y_sd_max = gp.predict(x_sd_max)[0]

        if False:
            plt.figure()
            plt.plot(X, y, "k.")
            plt.plot(X_pred, mean_pred, "r-")
            plt.plot(X_pred, mean_pred + 2*np.sqrt(var_pred), "r--")
            plt.plot(X_pred, mean_pred - 2*np.sqrt(var_pred), "r--")
            plt.plot(x_mean_max, y_mean_max, "k*", markersize=15)
            plt.plot(x_sd_max, y_sd_max, "k*", markersize=15)
            plt.grid()
            plt.show()

    def test_matern_grad_input(self):
        np.random.seed(0)
        n = 10
        k = 1
        X = np.random.uniform(low=-1., high=1., size=(n, k))
        ls = np.ones((k, )) * 0.1
        kernel = MaternKernel(1., ls, 0.)
        C = kernel.cov(X)
        y = np.random.multivariate_normal(np.zeros((n, )), C)
        gp = GaussianProcess(kernel)
        gp.fit(X, y)
        n_pred = 100
        X_pred = np.atleast_2d(np.linspace(-1., 1., num=n_pred)).T
        mean_pred, var_pred = gp.predict(X_pred)

        x_mean_max = np.zeros((1, k)) - 0.2
        x_sd_max = np.zeros((1, k)) + 0.5
        for i in range(1000):
            x_mean_max += 0.001 * gp.grad_input(x_mean_max)[0]
            x_sd_max += 0.001 * gp.grad_input(x_sd_max)[1]
        y_mean_max = gp.predict(x_mean_max)[0]
        y_sd_max = gp.predict(x_sd_max)[0]

        if False:
            plt.figure()
            plt.plot(X, y, "k.")
            plt.plot(X_pred, mean_pred, "r-")
            plt.plot(X_pred, mean_pred + 2*np.sqrt(var_pred), "r--")
            plt.plot(X_pred, mean_pred - 2*np.sqrt(var_pred), "r--")
            plt.plot(x_mean_max, y_mean_max, "k*", markersize=15)
            plt.plot(x_sd_max, y_sd_max, "k*", markersize=15)
            plt.grid()
            plt.show()


if __name__ == "__main__":
    unittest.main()
