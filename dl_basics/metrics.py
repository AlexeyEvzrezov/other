import torch


def accuracy(y_hat, y):
    """
    y_hat: predicted label probabilities of shape (n_objects, n_labels)
    y: true labels for every object of shape (n_objects,)
    :return: cross entropy loss of shape (n_labelsв,)
    """
    pred_class = y_hat.argmax(axis=1)
    res = pred_class.type(y.dtype) == y
    pass