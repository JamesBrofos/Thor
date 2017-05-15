from functools import reduce
from .abstract_kernel import AbstractKernel


class SumKernel(AbstractKernel):
    """Sum Kernel Class

    The sum of two kernels is also a kernel. This class allows a list of kernels
    to be combined by taking their sum. This class is particularly useful for
    creating "noisy" versions of common kernels such as the Matern and squared
    exponential kernels.

    If a noisy version of a kernel is desired, notice that the noise component
    is included as a separate input argument to this class. This is because we
    need to differentiate variation of the mean from variation of the target
    variable. This class supports this distinction explicitly; this
    functionality will manifest in sampling procedures for the target variable.

    Parameters:
        kernels (list): A list of kernel objects. Notice that we can also have a
            list consisting of a single element, if desired. These kernels will
            be combined through their sum.
        noise_kernel (NoiseKernel): The noise kernel that will introduce
            stochasticity into the target variable of prediction as well as
            generating stochastic samples from the posterior distribution of the
            probabilistic model.
    """
    def __init__(self, kernels, noise_kernel):
        """Initialize parameters of the sum kernel object."""
        self.kernels = kernels
        self.noise_kernel = noise_kernel

    @property
    def parameters(self):
        """Implementation of abstract base class property.

        Returns the list of parameters of the kernel. Notice that this is the
        tuple consisting of all of the parameters of the kernels to be added
        together in addition to the parameters of the noise kernel.
        """
        return reduce(
            lambda a, b: a+b, [k.parameters for k in self.kernels]
        ) + self.noise_kernel.parameters

    def cov(self, X, Y=None, include_noise=True):
        """Implementation of abstract base class method."""
        K = reduce(
            lambda K, H: K+H, [k.cov(X, Y) for k in self.kernels]
        )
        if include_noise:
            return K + self.noise_kernel.cov(X, Y)
        else:
            return K

    def grad_input(self, x, Y):
        """Implementation of abstract base class method."""
        return reduce(
            lambda K, H: K+H, [k.grad_input(x, Y) for k in self.kernels]
        )

    def grad_params(self, X):
        """Implementation of abstract base class method."""
        grads = {
            p: v for k in self.kernels for p, v in k.grad_params(X).items()
        }
        for p, v in self.noise_kernel.grad_params(X).items():
            grads[p] = v

        return grads
