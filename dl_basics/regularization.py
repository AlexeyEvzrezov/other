import torch


def l2_reg(w):
    return torch.sum(w ** 2) # / 2


def dropout(layer, p_drop):
    assert 0 <= p_drop <= 1
    if p_drop == 0:
        return layer
    if p_drop == 1:
        return torch.zeros_like(layer)
    mask = (torch.rand(layer.shape) > p_drop).float()
    return mask * layer / (1.0 - p_drop)
