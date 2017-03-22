import unittest
import numpy as np
from thor import kernels


class KernelsTest(unittest.TestCase):
    def test_abstract_kernel(self):
        """Test to ensure that the fundamentals of the kernels are working as
        expected.
        """
        # Create random matrices.
        n, k = 4, 3
        X = np.random.normal(size=(n, k))
        Y = np.random.normal(size=(n, k))
        # Create kernel.
        length_scales = np.ones((k, ))
        k = kernels.abstract_kernel.AbstractKernel(1., length_scales, 0.)
        # Compute distances.
        D_X = k.pairwise_distances(X, X)
        D_X_Y = k.pairwise_distances(X, Y)

    def test_squared_exponential(self):
        pass

    def test_grad_squared_exponential(self):
        np.random.seed(0)
        n = 10
        k = 2
        ls = np.ones((k, ))
        x = np.random.normal(size=(k, ))
        Y = np.random.normal(size=(n, k))
        k = kernels.SquaredExponentialKernel(1., ls, 0.)
        k.grad_input(x, Y)


if __name__ == "__main__":
    unittest.main()
