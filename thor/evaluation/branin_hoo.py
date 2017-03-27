import numpy as np


def branin_hoo(x):
    # Unfurl the input (so that we don't have to deal with two-dimensional
    # indexing)
    x = x.ravel()
    x1 = x[0]
    x2 = x[1]

    t = 1. / (8.*np.pi)
    s = 10.
    r = 6.
    c = 5. / np.pi
    b = 5.1 / (4.*np.pi ** 2)
    a = 1.
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1. - t) * np.cos(x1)

    y = term1 + term2 + s

    return y
