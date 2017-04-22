from .gaussian_process import GaussianProcess
from .student_process import StudentProcess
from .neural_network import BayesianNeuralNetwork


model_dict = {
    "gaussian_process": GaussianProcess,
    "student_t_process": StudentProcess
}
