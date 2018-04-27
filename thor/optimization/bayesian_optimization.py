import time
import numpy as np
from sif.models import GaussianProcess
from sif.kernels import MaternKernel
from sif.samplers import EllipticalSliceSampler
from sif.acquisitions import (
    ExpectedImprovement, ImprovementProbability, UpperConfidenceBound, ThompsonSampling
)


acq_dict = {
    "expected_improvement": ExpectedImprovement,
    "improvement_probability": ImprovementProbability,
    "upper_confidence_bound": UpperConfidenceBound,
    "thompson_sampling": ThompsonSampling
}


class BayesianOptimization:
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
            length_scales, amplitude, prior_mean = (
                np.exp(f[:k]), np.exp(f[k]), f[-1]
            )
            noise_level = np.exp(f[-2]) if len(f) == k + 3 else 1e-6
            gp = GaussianProcess(
                MaternKernel(length_scales, amplitude), noise_level, prior_mean
            )
            gp.fit(X, y)
            return gp.log_likelihood

        # Determine the number of burn-in iterations of elliptical slice
        # sampling to perform.
        burnin = 100 * k
        # Now use an elliptical slice sampler to draw samples from the Gaussian
        # process posterior mean function given samples of the Gaussian process
        # hyperparameters. In this example, we are sampling the kernel
        # amplitude, its length scales (of which there is only one since this
        # is a one-dimensional example), and the noise level of the process. We
        # use relatively uninformative priors.
        #
        # TODO: This inner function can be moved outside? It's input would need
        # to be `dim`, `n_models`, `burnin`, and `log_likelihood_func`.
        def elliptical_slice_sampling(dim):
            """Performs elliptical slice sampling on the Gaussian process
            marginal likelihood. This function takes an input `dim` which
            determines the number of parameters for which posterior parameters
            will be generated. In practice, this will only vary by one
            corresponding to noise-free or noisy Gaussian processes.
            """
            mean = np.zeros((dim, ))
            covariance = np.diag(np.ones((dim, )) * 5.)
            sampler = EllipticalSliceSampler(mean, covariance, log_likelihood_func)
            return sampler.sample(n_models, burnin)

        try:
            samples = elliptical_slice_sampling(k + 2)
            noiseless = True
            print("Successfully estimated noiseless Gaussian process model.")
        except:
            print(
                "Failed to estimate a noiseless Gaussian process. Reverting to "
                "non-zero noise model."
            )
            samples = elliptical_slice_sampling(k + 3)
            noiseless = False
        samples[:, :-1] = np.exp(samples[:, :-1])

        # Now create an individual Gaussian process model for each setting of
        # the kernel hyperparameters. This will allow us to integrate over the
        # model uncertainty in order to take into account different
        # interpretations of the data.
        models = []
        for i in range(n_models):
            length_scales, amplitude = samples[i, :k], samples[i, k]
            prior_mean = samples[i, -1]
            noise_level = 1e-6 if noiseless else samples[i, -2]
            gp = GaussianProcess(
                MaternKernel(length_scales, amplitude),
                noise_level,
                prior_mean
            )
            gp.fit(X, y)
            models.append(gp)

        return models

    def recommend(
            self,
            X,
            y,
            X_pending,
            model_class,
            n_models,
            acquisition,
            integrate_acq
    ):
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
            acquisition (string): A string specifying which acquisition function
                should be used. The permissible values are the keys in the
                dictionary `acq_dict`.
        """
        # Extract data size.
        n, k = X.shape
        # Transform original inputs into the unit hypercube.
        X = self.space.transform(X)
        # Estimate the probabilistic surrogate model.
        start_time = time.time()
        models = self.__fit_surrogate(X, y, model_class, n_models)
        print("Time elapsed to fit models: {:.4f}".format(time.time() - start_time))

        # Create fantasy observations for the pending values.
        if X_pending is not None:
            # Sample from the predictive posterior.
            n_pending = X_pending.shape[0]
            X_pending = self.space.transform(X_pending)
            y_pending = np.random.choice(models).sample(X_pending, target=True)
            # Retrain Gaussian process.
            X = np.vstack((X, X_pending))
            y = np.append(y, y_pending)
            try:
                for i in range(n_models):
                    models[i].fit(X, y)
            except:
                print("Unable to refit models with fantasy observations. Resampling hyperparameters.")
                models = self.__fit_surrogate(X, y, model_class, n_models)

        # Construct acquisition function and compute a recommendation from the
        # Bayesian optimization algorithm.
        start_time = time.time()
        A = acq_dict[acquisition]
        if integrate_acq:
            recs = A(models).select()[0]
        else:
            n_models = len(models)
            recs = np.zeros((n_models, k))
            for i, mod in enumerate(models):
                mod.fit(X, y)
                recs[i] = A(mod).select()[0]
                X = np.vstack((X, recs[i]))
                y = np.append(y, mod.sample(recs[i], target=True))

        print("Time elapsed to select recommendation with {}: {:.4f}".format(acquisition, time.time() - start_time))

        # Make sure that the recommendation really is in the correct interval.
        # This is by assumption the unit hypercube. Sometimes we may encounter
        # slightly negative values, which will be clipped to zero by this
        # method.
        return np.clip(recs, 0., 1.)
