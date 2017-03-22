import numpy as np
import unittest
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from thor.evaluation import branin_hoo, hartmann_6, camelback


class EvaluationTest(unittest.TestCase):
    def test_branin_hoo(self):
        n = 100
        r = np.linspace(0., 1., num=n)
        R = np.meshgrid(r, r)
        X, Y = R[0], R[1]
        Z = np.hstack((
            np.atleast_2d(X.ravel()).T,
            np.atleast_2d(Y.ravel()).T,
        ))
        # Compute Branin-Hoo function.
        y = np.zeros(X.shape)
        for i in range(n):
            for j in range(n):
                v = np.array([X[i, j], Y[i, j]])
                y[i, j] = branin_hoo(v)

        print("Discretized minimum:\t{}".format(y.min()))

        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, y)
            plt.show()

    def test_hartmann_6(self):
        x_opt = np.array([
            0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573
        ])
        self.assertAlmostEqual(hartmann_6(x_opt), -3.32236801139)

    def test_camelback(self):
        x_opt = np.array([0.0898, -0.7126])
        self.assertAlmostEqual(camelback(x_opt), -1.0316284229)
        self.assertAlmostEqual(camelback(-x_opt), -1.0316284229)

if __name__ == "__main__":
    unittest.main()
