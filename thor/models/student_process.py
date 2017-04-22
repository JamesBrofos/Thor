import numpy as np
import scipy.linalg as spla
from scipy.special import gammaln, digamma
from .abstract_process import AbstractProcess
from ..kernels.kernel_parameter import KernelParameter


class StudentProcess(AbstractProcess):
    """Student-t Process Class"""
    def __init__(self, kernel, nu=10):
        """Initialize the parameters of the Student-t process object."""
        super(StudentProcess, self).__init__(kernel)
        self.nu = KernelParameter("nu", nu, (2. + 1e-6, 30.))

    def log_likelihood(self):
        """Implementation of abstract base class method."""
        n = self.X.shape[0]
        return (
            -0.5 * n * np.log((self.nu.value[0] - 2) * np.pi) -
            np.sum(np.log(np.diag(self.L))) +
            gammaln((self.nu.value[0] + n) / 2.) - gammaln(self.nu.value[0] / 2.) -
            (self.nu.value[0] + n) / 2. * np.log(1. + self.beta / (self.nu.value[0] - 2.))
        ) / n

    def grad_log_likelihood(self):
        """Extension of base class method."""
        ll_grad = super(StudentProcess, self).grad_log_likelihood()
        n = self.X.shape[0]
        ll_grad[self.nu.name] = (
            -n / (2.*(self.nu.value[0] - 2.)) +
            digamma((self.nu.value[0] + n) / 2.) - digamma(self.nu.value[0] / 2.) -
            0.5 * np.log(1. + self.beta / (self.nu.value[0] - 2.)) +
            0.5 * ((self.nu.value[0] + n) * self.beta) / (
                (self.nu.value[0] - 2.) ** 2 + self.beta * (self.nu.value[0] - 2.)
            )
        )
        return ll_grad

    def sample(self, X):
        """Extension of base class method."""
        mean, cov = super(StudentProcess, self).sample(X)
        Y = np.random.multivariate_normal(np.zeros((cov.shape[0], )), cov)
        U = np.random.chisquare(self.nu.value[0])
        return mean + Y * np.sqrt(self.nu.value[0] / U)

    @property
    def parameters(self):
        """Implementation of abstract base class property."""
        return tuple(list(self.kernel.parameters) + [self.nu])

    @property
    def grad_prefactor(self):
        """Implementation of abstract base class property."""
        n = self.X.shape[0]
        return (self.nu.value[0] + n) / (self.nu.value[0] + self.beta - 2.)

    @property
    def predict_prefactor(self):
        """Implementation of abstract base class property."""
        n = self.X.shape[0]
        return (self.nu.value[0] + self.beta - 2.) / (self.nu.value[0] + n  - 2.)
