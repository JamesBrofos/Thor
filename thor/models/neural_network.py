import scipy.linalg as spla
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class BayesianLinearRegression(object):
    """Bayesian Linear Regression Class

    The Bayesian linear regression class allows for structured inference with
    well-characterized uncertainties for linear models. The Bayesian linear
    regression algorithm assumes a zero-mean prior over the covariates, but
    permits an arbitrary (diagonal) prior covariance.

    Parameters:
        prior_cov (numpy array): A numpy array representing the diagonal
            elements of a prior covariance matrix.

    Attributes:
        prior_proc (numpy array): A numpy matrix representing the inverse of the
            prior covariance. That is, the precision matrix.
        cov (numpy array): The posterior covariance of coefficients.
        mu (numpy array): The posterior mean of the coefficients. Together with
            the posterior covariance, this allows one to sample from the
            distribution of linear coefficients using a multivariate normal.
        sigma_sq (float): The noise variance under the posterior linear model.
    """
    def __init__(self, prior_cov):
        """Initialize the parameters of the Bayesian linear regression object.
        """
        # For computational simplicity, use the precision matrix rather than the
        # covariance matrix.
        self.prior_prec = np.diag(prior_cov ** -1)

    def fit(self, X, y):
        """Fit the Bayesian linear regression model leveraging the available
        data.

        Parameters:
            X (numpy array): A two-dimensional numpy array representing the
                matrix of covariates. Note that if a bias term is expressly
                desired, it must be included in the design matrix.
            y (numpy array): A matrix of target variables to predict.
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

        Parameters:
            X_pred (numpy array): A two-dimensional matrix of covariates for
                which inference is to be performed.
            covariance (boolean, optional): If true, the full covariance matrix
                of predictions will be returned as well as the posterior mean.
                If false, then the standard deviation for each point in the
                prediction design matrix will be returned instead. Note that in
                the latter case, it will be the standard deviation of the mean,
                not the standard deviation of the targets.

        Returns:
            A tuple containing the posterior mean at each point where prediction
                was done and either the full covariance matrix of the
                predictions under the posterior or the standard deviation at
                each point.
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

        Parameters:
            X (numpy array): A two-dimensional design matrix representing the
                inputs where a sample of the target variables under the
                posterior should be drawn.

        Returns:
            A numpy array representing one random draw from the posterior of the
                target variables.
        """
        n, k = X.shape
        m, C = self.predict(X, covariance=True)
        return np.random.multivariate_normal(m, C)


class BayesianNeuralNetwork(object):
    """Bayesian Neural Network Class

    The Bayesian neural network implements a neural network whose weights and
    biases are estimated by gradient ascent on the squared error loss function.
    Furthermore, as a post-hoc operation, a Bayesian linear regression model is
    applied to the learned basis representations.

    Parameters:
        n_epochs (int): The number of training gradient descent epochs to
            perform. Each gradient descent step is calculated on one-tenth of
            the dataset.
        uncertainty (float): The prior variance over the coefficients. Setting
            it to a small value is akin to expressing a large degree of
            uncertainty, whereas a large value is asymptotically equivalent to
            maximum likelihood linear regression.
    """
    def __init__(self, n_epochs, uncertainty=100.0):
        """Initialize the parameters of the Bayesian neural network object."""
        self.n_epochs = n_epochs
        self.uncertainty = uncertainty

    def fit(self, X, y):
        """Fit a neural network to the available training data and apply a
        post-hoc Bayesian linear regression layer.

        Parameters:
            X (numpy array): A two-dimensional numpy array representing the
                matrix of covariates. Note that if a bias term is expressly
                desired, it must be included in the design matrix.
            y (numpy array): A matrix of target variables to predict.
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
        prior_cov = np.ones((n_hidden, )) * self.uncertainty
        self.bayes_reg = BayesianLinearRegression(prior_cov)
        self.bayes_reg.fit(R, self.y)

    def predict(self, X_pred):
        """Predict using the neural network class at the provided inputs. This
        method has two steps where first the input is transformed into a basis
        vector representation using the neural network and then, using those
        basis vectors, is mapped into the target space using Bayesian linear
        regression.

        Parameters:
            X_pred (numpy array): A two-dimensional matrix of covariates for
                which inference is to be performed.

        Returns:
            A tuple containing the posterior mean at each point where prediction
                was done and either the full covariance matrix of the
                predictions under the posterior or the standard deviation at
                each point.
        """
        # Obtain the representation at the final hidden layer.
        R = self.proxy.predict(X_pred)
        return self.bayes_reg.predict(R)

    def sample(self, X):
        """Sample the target variable using the post-hoc Bayesian linear
        regression layer.

        Parameters:
            X (numpy array): A two-dimensional design matrix representing the
                inputs where a sample of the target variables under the
                posterior should be drawn.

        Returns:
            A numpy array representing one random draw from the posterior of the
                target variables.
        """
        return self.bayes_reg.sample(self.proxy.predict(X))

