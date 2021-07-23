import torch


def softmax(X, axis):
    X_exp = torch.exp(X)
    partition = X_exp.sum(axis=axis, keepdims=True)
    return X_exp / partition


def linear_with_softmax(X, W, b):
    logits = X.reshape(-1, W.shape[0]) @ W + b
    return softmax(logits, axis=1)


def cross_entropy(y_hat, y):
    """
    :param y_hat: predicted label probabilities of shape (n_objects, n_labels)
    :param y: true labels for every object of shape (n_objects,)
    :return: cross entropy loss of shape (n_labelsв,)
    """
    return -torch.log(y_hat[range(y_hat.shape[0]), y])
