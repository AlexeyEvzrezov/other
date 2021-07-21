import torch
import torchvision
from torch.utils import data
from torchvision import transform


num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
