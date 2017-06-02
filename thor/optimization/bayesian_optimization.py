import numpy as np
from thor.acquisitions import acq_dict
from thor.kernels import MaternKernel, NoiseKernel, SumKernel
from thor.models.abstract_process import fit_marginal_likelihood
from thor.models import BayesianNeuralNetwork, StudentProcess, GaussianProcess


class BayesianOptimization(object):
    """Bayesian Optimization Class

    Bayesian optimization seeks to maximizes a latent objective function by
    intelligently fine-tuning model parameters. These parameters are often
    called "hyperparameters" since they are not learned directly from the data,
    but must typically be set a priori by the user. Example use-cases of
    Bayesian optimization include tuning the regularization weight for each
    layer in a neural network, selecting the learning rate parameter in a
    gradient descent algorithm, and choosing the slack weight in an support
    vector machine.

    Thor is a library that supports Bayesian optimization through both an API
    and an intuitive user interface. This software library performs the
    mathematics behind Bayesian optimization, such as estimating the underlying
    surrogate probabilistic model, handling pending recommendations, and
    maximizing the acquisition function with respect to inputs to your machine
    learning model.

    Parameters:
        experiment (Experiment): The experiment object represents an object
            stored in a relational database on Thor Server. This allows
            experiments to be persistent across sessions. This object will
            specify the name of the experiment, the acquisition function to be
            used, the date the experiment was created, the user who created the
            experiment, as well as the associated observations of the latent
            objective function and the tunable dimensions of the problem.
        space (Space): This object represents the space of tunable parameters of
            the machine learning problem. The space is defined by box
            constraints, consisting of a minimum and maximum along each
            dimension.
    """
    def __init__(self, experiment, space):
        """Initialize parameters of the Bayesian optimization object."""
        self.experiment = experiment
        self.space = space

    def __fit_surrogate(self, X, y, model_class, n_model_iters):
        """This function actually returns the model after its parameters have
        been estimated. For Gaussian processes, fitting the model means
        estimating the length scales, amplitude, and noise level of the kernel
        function. For neural networks, the weights and biases of three network
        layers must be optimized.
        """
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

        Parameters:
            X (numpy array): A two-dimensional numpy array of observed inputs
                from the machine learning system. Each row represents a
                configuration of model hyperparameters.
            y (numpy array): A vector representing observed values of the latent
                metric to be maximized.
            X_pending (numpy array): A two-dimensional numpy array where each
                row represents a configuration of model hyperparameters that
                have not yet been computed. This distinguishes the entries of
                `X_pending` from the entries of `X`.
            model_class (GaussianProcess or BayesianNeuralNetwork): A
                probabilistic model used by Thor to interpolate the latent
                objective function with respect to observed hyperparameter
                configurations. If a Gaussian process object, then the runtime
                will be cubic in the number of observations; on the other hand,
                if a Bayesian neural network object, then the runtime will be
                only linear in the number of observations.
            n_model_iters (int): If a Gaussian process object is used, this is
                the number of random restarts used to estimate the kernel
                parameters. If a Bayesian neural network is used, then this is
                the number of training epochs to perform divided by one hundred.
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

