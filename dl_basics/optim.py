import torch


class SGD:
    """Minibatch stochastic gradient descent.
    It is supposed that loss is calculated as a mean over the minibatch.
    params: iterable containing weights and biases, e.g. [W, b]
    lr: learning rate
    """

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad
                param.grad.zero_()
