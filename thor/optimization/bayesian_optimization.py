import numpy as np
from sif.acquisitions import ExpectedImprovement, ImprovementProbability
from sif.models import GaussianProcess
from sif.kernels import MaternKernel
from sif.samplers import EllipticalSliceSampler

acq_dict = {
    "expected_improvement": ExpectedImprovement,
    "improvement_probability": ImprovementProbability,
}


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

    def __fit_surrogate(self, X, y, model_class, n_models):
        """This function actually returns the model after its parameters have
        been estimated. For Gaussian processes, fitting the model means
        estimating the length scales, amplitude, and noise level of the kernel
        function. For neural networks, the weights and biases of three network
        layers must be optimized.
        """
        # Extract data size.
        n, k = X.shape

        def log_likelihood_func(f):
            """Create the Gaussian process object and a function to express the
            log-likelihood of the data under a given specification of the
            Gaussian process hyperparameters. Instead of using a squared
            exponential kernel for the Gaussian process covariance, one might
            instead consider using the Matern-5/2 kernel, which produces less
            smooth interpolations.
            """
            gp = GaussianProcess(
                MaternKernel(np.exp(f[:k]), np.exp(f[-3])), np.exp(f[-2]), f[-1]
            )
            gp.fit(X, y)
            return gp.log_likelihood

        # Now use an elliptical slice sampler to draw samples from the Gaussian
        # process posterior mean function given samples of the Gaussian process
        # hyperparameters. In this example, we are sampling the kernel
        # amplitude, its length scales (of which there is only one since this is
        # a one-dimensional example), and the noise level of the process. We use
        # relatively uninformative priors.
        mean = np.zeros((k + 3, ))
        covariance = np.diag(np.ones((k + 3, )) * 5.)
        sampler = EllipticalSliceSampler(mean, covariance, log_likelihood_func)
        samples = sampler.sample(n_models)
        samples[:, :-1] = np.exp(samples[:, :-1])

        # Now create an individual Gaussian process model for each setting of
        # the kernel hyperparameters. This will allow us to integrate over the
        # model uncertainty in order to take into account different
        # interpretations of the data.
        models = []
        for i in range(n_models):
            gp = GaussianProcess(
                MaternKernel(samples[i, :k], samples[i, -3]),
                samples[i, -2],
                samples[i, -1]
            )
            gp.fit(X, y)
            models.append(gp)

        return models

    def recommend(self, X, y, X_pending, model_class, n_models):
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
            n_models (int): If a Gaussian process object is used, this is the
                number of hyperparameters from the posterior to sample.
        """
        # Extract data size.
        n, k = X.shape
        # Transform original inputs into the unit hypercube.
        for i in range(n):
            X[i] = self.space.transform(X[i])
        # Construct acquisition function.
        acq = acq_dict[self.experiment.acq_func.name]
        # Estimate the probabilistic surrogate model.
        models = self.__fit_surrogate(X, y, model_class, n_models)

        # Create fantasy observations for the pending values.
        if X_pending is not None:
            # Sample from the predictive posterior.
            n_pending = X_pending.shape[0]
            for i in range(n_pending):
                X_pending[i] = self.space.transform(X_pending[i])
            y_pending = np.random.choice(models).sample(X_pending)
            # Retrain Gaussian process.
            X = np.vstack((X, X_pending))
            y = np.append(y, y_pending)
            try:
                for i in range(n_models):
                    models[i].fit(X, y)
            except:
                models = self.__fit_surrogate(X, y, model_class, n_models)

        # Compute a recommendation from the Bayesian optimization algorithm.
        return self.space.invert(acq(models).select()[0])

