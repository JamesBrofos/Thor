import numpy as np


class Space(object):
    """Space Class

    The space class refers to the domain, typically a subset of the real numbers
    in the corresponding number of dimensions, over which to optimize a latent
    function.
    """
    def __init__(self, dimensions):
        """Initialize parameters of the space object."""
        self.dimensions = dimensions
        self.n_dims = len(self.dimensions)

    def sample(self, n_samples=1):
        """Sample from the uniform distribution applied to each dimension of the
        space. Notice that this function will produce random samples from the
        original space, not the unit hypercube.
        """
        v = np.zeros((n_samples, self.n_dims))
        for i in range(n_samples):
            for j in range(self.n_dims):
                v[i, j] = self.dimensions[j].sample()

        return v

    def transform(self, original_input):
        """Apply the transformation formula along each dimesnion in order to
        create a point in the unit hypercube based on a one-to-one mapping from
        the original search space.
        """
        t = np.zeros(self.n_dims)
        for i in range(self.n_dims):
            t[i] = self.dimensions[i].transform(original_input[i])

        return t

    def invert(self, transformed_input):
        """Apply the inversion formula along each dimension in order to
         transform a point in the unit hypercube to a point in the original
        search space.
        """
        t = np.zeros(self.n_dims)
        for i in range(self.n_dims):
            t[i] = self.dimensions[i].invert(transformed_input[i])

        return t

