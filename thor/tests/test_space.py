import unittest
import numpy as np
from thor.space import LinearDimension, IntegerDimension, Space


class SpaceTest(unittest.TestCase):
    def test_space(self):
        dims = [
            LinearDimension(-3., 5.),
            IntegerDimension(100, 550),
            LinearDimension(0., 1.)
        ]
        s = Space(dims)
        print(s.sample(100))

    def test_linear_dimension(self):
        low, high = -5., 3.
        d = LinearDimension(low, high)

        for _ in range(10000):
            s = d.sample()
            self.assertTrue(s >= low and s <= high)
            u = np.random.uniform()
            inverted = d.invert(u)
            self.assertTrue(inverted >= low and inverted <= high)
            original = np.random.uniform(low, high)
            transformed = d.transform(original)
            self.assertTrue(transformed >= 0. and transformed <= 1.)

    def test_integer_dimension(self):
        low, high = -5, 3
        d = IntegerDimension(low, high)
        freqs = {i: 0 for i in range(low, high+1)}
        n_trials = 1000

        for _ in range(n_trials):
            s = d.sample()
            self.assertTrue(s >= low and s <= high)
            u = np.asarray(np.random.uniform())
            inverted = d.invert(u)
            freqs[inverted] += 1
            self.assertTrue(inverted >= low and inverted <= high)
            original = np.asarray(np.random.uniform(low, high))
            transformed = d.transform(original)
            self.assertTrue(transformed >= 0. and transformed <= 1.)

        for k in freqs:
            freqs[k] /= n_trials

        print("Integer frequencies:")
        print(freqs)


if __name__ == "__main__":
    unittest.main()
