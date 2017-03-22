import numpy as np


def hartmann_6(x, original=True):
    # Unfurl the input.
    x = x.ravel()
    # Initialize matrices for the Hartmann function.
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10**(-4) * np.array(
        [[1312, 1696, 5569, 124, 8283, 5886],
         [2329, 4135, 8307, 3736, 1004, 9991],
         [2348, 1451, 3522, 2883, 3047, 6650],
         [4047, 8828, 8732, 5743, 1091, 381]]
    )

    # Initialize a rolling sum for the Hartmann 6-D function.
    outer = 0.
    # Iterative construct all the terms in the Hartmann function.
    for i in range(4):
        inner = 0.
        for j in range(6):
            x_j = x[j]
            A_ij = A[i, j]
            P_ij = P[i, j]
            inner += A_ij * (x_j - P_ij) ** 2

        outer += alpha[i] * np.exp(-inner)

    # Compute the Hartmann 6-D function.
    y = -outer if original else -(2.58 + outer) / 1.94

    return y
