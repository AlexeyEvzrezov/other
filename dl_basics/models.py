import torch

from dl_basics.funcs import softmax


class SoftmaxRegression(object):
    def __init__(self, in_features, out_features):
        self.W = torch.normal(0, 0.01, (in_features, out_features), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.params = [self.W, self.b]

    def __call__(self, X):
        logits = X.reshape(-1, self.W.shape[0]) @ self.W + self.b
        return softmax(logits)
