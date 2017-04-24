import numpy as np
from thor.acquisitions import acq_dict
from thor.kernels import MaternKernel, NoiseKernel, SumKernel
from thor.models.abstract_process import fit_marginal_likelihood
from thor.models import BayesianNeuralNetwork, StudentProcess, GaussianProcess


class BayesianOptimization(object):
    """Bayesian Optimization Class"""
    def __init__(self, experiment, space):
        """Initialize parameters of the Bayesian optimization object."""
        self.experiment = experiment
        self.space = space

    def __fit_surrogate(self, X, y, model_class, n_model_iters):
        # Extract data size.
        n, k = X.shape
        # Change behavior depending on whether or not the experiment is large
        # scale.
        n_shift = 500

        # Nota bene: We will select in this piece of code the number of random
        # restarts used in the estimation of the Gaussian process (if indeed we
        # are using a Gaussian process surrogate model), which is a function of
        # the number of dimensions.
        if n <= n_shift:
            # Try to fit a noiseless model first, and then switch to a noisy
            # model if the mathematics is numerically unstable.
            dom_kernel = MaternKernel(np.nan, np.full((k, ), np.nan))
            noise_kernel = NoiseKernel(np.nan)
            sum_kernel = SumKernel([dom_kernel], noise_kernel)
            try:
                model = fit_marginal_likelihood(
                    X, y, n_model_iters, dom_kernel, model_class
                )
            except UnboundLocalError:
                model = fit_marginal_likelihood(
                    X, y, n_model_iters, sum_kernel, model_class
                )
        else:
            # Use Bayesian neural networks for large-scale problems.
            # Nota bene: We are multiplying the number of model restarts by one
            # thousand in order to produce the number of training epochs that
            # should be performed.
            model = BayesianNeuralNetwork(n_model_restarts * 100)
            model.fit(X, y)

        return model

    def recommend(self, X, y, X_pending, model_class, n_model_iters):
        """Choose points to evaluate from the parameter space based on Bayesian
        optimization. This function uses multiple random restarts in the unit
        hypercube in order to identify local maxima of the acquisition function.
        """
        # Extract data size.
        n, k = X.shape
        # Transform original inputs into the unit hypercube.
        for i in range(n):
            X[i] = self.space.transform(X[i])
        # Construct acquisition function.
        acq = acq_dict[self.experiment.acq_func.name]
        # Estimate the probabilistic surrogate model.
        model = self.__fit_surrogate(X, y, model_class, n_model_iters)

        # Create fantasy observations for the pending values.
        if X_pending is not None:
            # Sample from the predictive posterior.
            n_pending = X_pending.shape[0]
            for i in range(n_pending):
                X_pending[i] = self.space.transform(X_pending[i])
            y_pending = model.sample(X_pending)
            # Retrain Gaussian process.
            X = np.vstack((X, X_pending))
            y = np.append(y, y_pending)
            try:
                model.fit(X, y)
            except:
                model = self.__fit_surrogate(X, y, model_class, n_model_iters)

        # Compute a recommendation from the Bayesian optimization algorithm.
        return self.space.invert(
            acq(model, self.experiment.acq_func).select()
        )

