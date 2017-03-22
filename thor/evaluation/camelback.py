import numpy as np


def camelback(x):
    # Unfurl input.
    x = x.ravel()
    y = x[0]
    z = x[1]
    # Compute and return the value of the camelback function at the specified
    # input.
    T1 = (4. - 2.1 * y**2 + (y**4) / 3.) * y**2
    T2 = y * z
    T3 = (-4. + 4. * z**2) * z**2

    return T1 + T2 + T3

