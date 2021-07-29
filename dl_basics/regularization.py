import torch


def l2_reg(w):
    return torch.sum(w ** 2) # / 2
