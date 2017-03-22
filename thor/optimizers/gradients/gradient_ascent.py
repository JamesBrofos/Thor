class GradientAscent(object):
    """Gradient Ascent Class"""
    def __init__(self, n_iters, learning_rate, grad_func):
        """Initialize the parameters of the gradient ascent object."""
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.grad_func = grad_func

    def optimize(self, x):
        """Optimize the input by hill-climbing with the gradient."""
        for i in range(self.n_iters):
            x += self.learning_rate * self.grad_func(x)

        # TODO: Do I need to return the input or is this an in-place transform?
        # return x
