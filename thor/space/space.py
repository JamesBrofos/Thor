import numpy as np


class Space:
    """Space Class

    The space class refers to the domain, typically a subset of the real numbers
    in the corresponding number of dimensions, over which to optimize a latent
    function.

    Parameters:
        dimensions (list): A list of dimension objects that define the space
            over which Bayesian optimization will be performed.

    Attributes:
        n_dims (integer): The number of dimensions.
    """
    def __init__(self, dimensions):
        """Initialize parameters of the space object."""
        self.dimensions = dimensions
        self.n_dims = len(self.dimensions)

    def sample(self, n_samples=1):
        """Sample from the uniform distribution applied to each dimension of the
        space. Notice that this function will produce random samples from the
        original space, not the unit hypercube.

        TODO: Should there be an internal variable for keeping track of which
        randomly generated Sobol point should be generated?

        Parameters:
            n_samples (integer, optional): The number of random samples to
                generate. By default, only a single sample will be generated.
        """
        v = np.zeros((n_samples, self.n_dims))
        for i in range(n_samples):
            for j in range(self.n_dims):
                v[i, j] = self.dimensions[j].sample()

        return v

    def transform(self, original_input):
        """Apply the transformation formula along each dimension in order to
        create a point in the unit hypercube based on a one-to-one mapping from
        the original search space.

        Parameters:
            original_input (numpy array): An array representing a location in
                the space. This is mapped into the unit hypercube by applying a
                transformation into the unit interval along each of the
                component dimensions.

        Returns:
            A point in the unit hypercube.
        """
        t = np.zeros(self.n_dims)
        for i in range(self.n_dims):
            t[i] = self.dimensions[i].transform(original_input[i])

        return t

    def invert(self, transformed_input):
        """Apply the inversion formula along each dimension in order to
         transform a point in the unit hypercube to a point in the original
        search space.

        Parameters:
            transformed_input (numpy array): An array representing a location in
                the unit hypercube. This is mapped into the space by applying an
                inversion to the original dimension along each axis of the unit
                hypercube.

        Returns:
            A point in the space of original inputs.
        """
        if transformed_input.ndim == 2:
            n = transformed_input.shape[0]
            t = np.zeros((n, self.n_dims))
            for i in range(n):
                t[i] = self.invert(transformed_input[i])
        else:
            t = np.zeros(self.n_dims)
            for i in range(self.n_dims):
                t[i] = self.dimensions[i].invert(transformed_input[i])
        return t

