import numpy as np
from thor.acquisitions import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ImprovementProbability,
    HedgeAcquisition
)
from thor.kernels import MaternKernel, NoiseKernel, SumKernel
from thor.models.tuning import fit_marginal_likelihood
from thor.models import BayesianNeuralNetwork


class BayesianOptimization(object):
    """Bayesian Optimization Class"""
    def __init__(self, experiment, space):
        """Initialize parameters of the Bayesian optimization object."""
        self.experiment = experiment
        self.space = space

    def construct_acquisition(self, model):
        acq = self.experiment.acq_func
        return {
            "expected_improvement": ExpectedImprovement,
            "improvement_probability": ImprovementProbability,
            "upper_confidence_bound": UpperConfidenceBound,
            "hedge": HedgeAcquisition
        }[acq.name](model, acq)

    def recommend(self, X, y, X_pending):
        """Choose points to evaluate from the parameter space based on Bayesian
        optimization. This function uses multiple random restarts in the unit
        hypercube in order to identify local maxima of the acquisition function.
        """
        # Extract data size.
        n, k = X.shape
        # Transform original inputs into the unit hypercube.
        for i in range(n):
            X[i] = self.space.transform(X[i])
        # Change behavior depending on whether or not the experiment is large
        # scale.
        n_shift = 500

        # Nota bene: We will select in this piece of code the number of random
        # restarts used in the estimation of the Gaussian process (if indeed we
        # are using a Gaussian process surrogate model), which is a function of
        # the number of dimensions.
        if n <= n_shift:
            # Create a Gaussian process probabilistic model for small-to-medium
            # sized experiments.
            pm = 0.
            n_restarts = 5 * k

            # Try to fit a noiseless model first, and then switch to a noisy
            # model if the mathematics is numerically unstable.
            # TODO: Can this be done more elegantly?
            try:
                kernel = MaternKernel(np.nan, np.full((k, ), np.nan))
                model = fit_marginal_likelihood(X, y, n_restarts, kernel, pm)
            except UnboundLocalError:
                dom_kernel = MaternKernel(np.nan, np.full((k, ), np.nan))
                noise_kernel = NoiseKernel(np.nan)
                kernel = SumKernel([dom_kernel], noise_kernel)
                model = fit_marginal_likelihood(X, y, n_restarts, kernel, pm)
        else:
            # Use Bayesian neural networks for large-scale problems.
            n_epochs = 1000
            model = BayesianNeuralNetwork(n_epochs)
            model.fit(X, y)

        # Create fantasy observations for the pending values.
        if X_pending:
            # Sample from the predictive posterior.
            n_pending = X_pending.shape[0]
            for i in range(n_pending):
                X_pending[i] = self.space.transform(X_pending[i])
            y_pending = model.sample(X_pending)
            # Retrain Gaussian process.
            X = np.vstack((X, X_pending))
            y = np.append(y, y_pending)
            model.fit(X, y)

        # Construct the acquisition function.
        acq = self.construct_acquisition(model)
        # Compute a recommendation from the Bayesian optimization algorithm.
        return self.space.invert(acq.maximize())

