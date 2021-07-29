import torch
from matplotlib import pyplot as plt


def plot_activation(fn):
    x = torch.arange(-3.0, 3.0, 0.01, requires_grad=True)
    y = fn(x)
    fn_name = fn.__name__
    plt.plot(x.detach(), y.detach(), label=fn_name)
    y.backward(torch.ones_like(x), retain_graph=True)
    plt.plot(x.detach(), x.grad, label=f'{fn_name} grad')
    plt.legend()
