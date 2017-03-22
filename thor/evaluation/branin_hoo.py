import numpy as np


def branin_hoo(x):
    # Unfurl the input (so that we don't have to deal with two-dimensional
    # indexing)
    x = x.ravel()
    # Rescale the input to be in the appropriate range.
    xbar = 15. * x[0] - 5.
    ybar = 15. * x[1]
    alpha = (
        ybar - 5.1 * xbar ** 2 / (4. * np.pi ** 2) + 5. * xbar / np.pi - 6.
    )
    beta = (10. - 10. / (8. * np.pi)) * np.cos(xbar)

    return (alpha ** 2 + beta - 44.81) / 51.95
