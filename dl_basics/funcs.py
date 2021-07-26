import numpy as np
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


def softmax(X, axis=1):
    X_exp = torch.exp(X)
    partition = X_exp.sum(axis=axis, keepdims=True)
    return X_exp / partition


def softmax_regression(X, W, b):
    logits = X.reshape(-1, W.shape[0]) @ W + b
    return softmax(logits)


def cross_entropy_loss(y_hat, y):
    """
    y_hat: predicted label probabilities of shape (n_objects, n_labels)
    y: true labels for every object of shape (n_objects,)
    :return: cross entropy loss of shape (n_labelsв,)
    """
    return -torch.log(y_hat[range(y_hat.shape[0]), y])


def accuracy(y_hat, y):
    pred_class = y_hat.argmax(axis=1)
    res = pred_class.type(y.dtype) == y
    return


def load_data(X, y, batch_size, shuffle=True):
    n_samples = len(X)
    ids = list(range(n_samples))
    if shuffle:
        np.random.shuffle(ids)
    for i in range(0, n_samples, batch_size):
        batch_ids = torch.tensor(ids[i:min(i + batch_size, n_samples)])
        yield X[batch_ids], y[batch_ids]


def train(model, loss_fn, dataloader, optimizer, n_epochs):
    for epoch in range(n_epochs):
        for X, y in dataloader:
            y_hat = model(X) # fix model with W, b
            loss = loss_fn(y_hat, y)
            loss.mean().backward()
            optimizer.step()
