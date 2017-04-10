import scipy.linalg as spla
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class BayesianLinearRegression(object):
    """Bayesian Linear Regression Class"""
    def __init__(self, prior_cov):
        """Initialize the parameters of the Bayesian linear regression object.
        """
        # For computational simplicity, use the precision matrix rather than the
        # covariance matrix.
        self.prior_prec = np.diag(prior_cov ** -1)

    def fit(self, X, y):
        """Fit the Bayesian linear regression model leveraging the available
        data.
        """
        P = X.T.dot(X) + self.prior_prec
        L = spla.cholesky(P, lower=True)
        self.mu = spla.cho_solve((L, True), X.T.dot(y))
        self.sigma_sq = np.mean((X.dot(self.mu) - y) ** 2)
        L_inv = spla.solve_triangular(L.T, np.eye(L.shape[0]))
        self.cov = self.sigma_sq * L_inv.dot(L_inv.T)

    def predict(self, X_pred, covariance=False):
        """Leverage the Bayesian posterior distribution to compute the predicted
        mean and variance of the Bayesian linear regression model given the
        training data.
        """
        mean = X_pred.dot(self.mu)
        C = X_pred.dot(self.cov).dot(X_pred.T)
        if covariance:
            cov = C + self.sigma_sq * np.eye(X_pred.shape[0])
            return mean, cov
        else:
            sd = np.sqrt(np.diag(C))
            return mean, sd

    def sample(self, X):
        """Sample from the target distribution under the Bayesian linear
        regression posterior.
        """
        n, k = X.shape
        m, C = self.predict(X, covariance=True)
        return np.random.multivariate_normal(m, C)


class BayesianNeuralNetwork(object):
    """Bayesian Neural Network Class"""
    def __init__(self, n_epochs):
        """Initialize the parameters of the Bayesian neural network object."""
        self.n_epochs = n_epochs

    def fit(self, X, y):
        """Fit a neural network to the available training data and apply a
        post-hoc Bayesian linear regression layer.
        """
        # Extract number of observations and number of input space features.
        self.X, self.y = X, y
        n, k = self.X.shape
        # Construct the neural network model.
        n_hidden = 50
        model = Sequential()
        model.add(Dense(n_hidden, input_dim=k, activation="tanh"))
        model.add(Dense(n_hidden, activation="tanh"))
        model.add(Dense(n_hidden, activation="tanh"))
        model.add(Dense(1, activation="linear"))
        # Compile model and train the model.
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=[]
        )
        model.fit(self.X, self.y, epochs=self.n_epochs, batch_size=int(n / 10))

        # Build a proxy model for the purposes of obtaining the representation
        # in the final hidden layer of the network.
        self.proxy = Sequential()
        for i in range(len(model.layers)-1):
            self.proxy.add(Dense(
                n_hidden,
                input_dim=k if i == 0 else None,
                activation="tanh",
                weights=model.layers[i].get_weights()
            ))

        # Fit the Bayesian linear regression model.
        R = self.proxy.predict(self.X)
        self.bayes_reg = BayesianLinearRegression(np.ones((n_hidden, )) * 100.)
        self.bayes_reg.fit(R, self.y)

    def predict(self, X_pred):
        """Predict using the neural network class at the provided inputs."""
        # Obtain the representation at the final hidden layer.
        R = self.proxy.predict(X_pred)
        return self.bayes_reg.predict(R)

    def sample(self, X):
        """Sample the target variable using the post-hoc Bayesian linear
        regression layer.
        """
        return self.bayes_reg.sample(self.proxy.predict(X))

