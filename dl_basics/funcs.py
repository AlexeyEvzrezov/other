import torch


def softmax(X, axis=1):
    X_exp = torch.exp(X)
    partition = X_exp.sum(axis=axis, keepdims=True)
    return X_exp / partition

            
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
