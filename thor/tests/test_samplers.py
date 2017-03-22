import unittest
import numpy as np
from thor.samplers import MultivariateNormalSampler


class SamplerTest(unittest.TestCase):
    def test_multivariate_normal_sampler(self):
        n_sample = 1000000
        k = 3
        mean = np.zeros((k, ))
        # Create a random matrix for the covariance.
        X = np.random.normal(size=(10, k))
        cov = X.T.dot(X)
        # Sample from the corresponding of the multivariate normal.
        mvn_sampler = MultivariateNormalSampler(n_sample, mean, cov)
        samples = mvn_sampler.sample()


if __name__ == "__main__":
    unittest.main()
