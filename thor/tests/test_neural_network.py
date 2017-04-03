import unittest
import numpy as np
import matplotlib.pyplot as plt
from thor.models import BayesianNeuralNetwork


class NeuralNetworkTest(unittest.TestCase):
    """Test Module for Bayesian Neural Network Class"""
    def test_bayesian_neural_network(self):
        # Create fake data.
        n = 1000
        X = np.random.uniform(size=(n, 1))
        mult = 15.
        y = np.sin(mult*X).ravel() + np.random.normal(size=(n, )) / 10.

        # Fit the neural network.
        n_epochs = 200
        net = BayesianNeuralNetwork(n_epochs)
        net.fit(X, y)

        # Predict using the neural network.
        n_pred = 1000
        X_pred = np.atleast_2d(np.linspace(0., 1., num=n_pred)).T
        mean, sd = net.predict(X_pred)
        # Sample from the predictive distribution over targets.
        y_sample = net.sample(X_pred)

        # Visualize.
        if True:
            plt.figure(figsize=(18, 6))
            plt.plot(X.ravel(), y, "k.")
            plt.plot(X_pred.ravel(), y_sample, "b.")
            plt.plot(X_pred.ravel(), mean, "r-")
            plt.plot(X_pred.ravel(), mean + 2*sd, "r--")
            plt.plot(X_pred.ravel(), mean - 2*sd, "r--")
            plt.grid()
            plt.show()


if __name__ == "__main__":
    unittest.main()
