import numpy as np
from .abstract_kernel import AbstractKernel


class MaternKernel(AbstractKernel):
    """Matern Kernel Class"""
    def __init__(self, amplitude, length_scale, noise_level, nu="5/2"):
        """Initialize parameters of the Matern kernel object."""
        super(MaternKernel, self).__init__(amplitude, length_scale, noise_level)
        self.nu = nu
        self.kernel_name = "Matern Kernel {}".format(self.nu)

    def cov(self, X, Y=None):
        """Implementation of abstract base class method."""
        if Y is None:
            Y = X
            noise_variance = self.noise_level * np.eye(X.shape[0])
        else:
            noise_variance = 0.
        # Compute the squared distances between inputs under the current length
        # scales.
        r_sq = self.pairwise_distances(X, Y)
        r = np.sqrt(r_sq)
        # Compute the Matern kernel.
        if self.nu == "1/2":
            k = np.exp(-r)
        elif self.nu == "3/2":
            k = (1. + np.sqrt(3.)*r) * np.exp(-np.sqrt(3.)*r)
        elif self.nu == "5/2":
            k = (1. + np.sqrt(5.)*r + 5./3.*r_sq) * np.exp(-np.sqrt(5.)*r)
        else:
            raise ValueError(
                "Invalid nu parameter in Matern kernel: {}".format(self.nu)
            )

        return self.amplitude * k + noise_variance

    def grad_input(self, x, Y):
        """Implementation of abstract base class method."""
        # TODO: This needs to be refactored and documented.
        diff = x - Y
        diff /= self.length_scales
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)

        sqrt_5_dist = np.sqrt(5.) * dist
        f2 = (5.0 / 3.0) * dist_sq
        f2 += sqrt_5_dist
        f2 += 1
        f = np.expand_dims(f2, axis=1)

        nzd_mask = dist != 0.0
        nzd = dist[nzd_mask]
        dist[nzd_mask] = np.reciprocal(nzd, nzd)

        dist *= np.sqrt(5.)
        dist = np.expand_dims(dist, axis=1)
        diff /= self.length_scales
        f1_grad = dist * diff
        f2_grad = (10.0 / 3.0) * diff
        f_grad = f1_grad + f2_grad

        sqrt_5_dist *= -1
        g = np.exp(sqrt_5_dist, sqrt_5_dist)
        g = np.expand_dims(g, axis=1)
        g_grad = -g * f1_grad
        return f * g_grad + g * f_grad


    def grad_params(self, X):
        """Implementation of abstract base class method."""
        # TODO: Needs to be refactored.
        D = (
            (np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0)) ** 2 /
            (self.length_scales ** 3)
        )
        if self.nu == "3/2":
            K_ls_grad = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
        elif self.nu == "5/2":
            tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
            K_ls_grad = (5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp))
        else:
            raise ValueError(
                "Invalid nu parameter in Matern kernel: {}".format(self.nu)
            )
        K_amp_grad = self.cov(X) - self.noise_level * np.eye(X.shape[0])
        return K_amp_grad, K_ls_grad


