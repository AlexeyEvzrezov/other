import torch


def cross_entropy_loss(y_hat, y):
    """
    y_hat: predicted label probabilities of shape (n_objects, n_labels)
    y: true labels for every object of shape (n_objects,)
    :return: cross entropy loss of shape (n_labelsв,)
    """
    return -torch.log(y_hat[range(y_hat.shape[0]), y])
