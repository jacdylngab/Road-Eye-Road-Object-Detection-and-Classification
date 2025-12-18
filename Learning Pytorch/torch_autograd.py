"""
torch.autograd is PyTorch's automatic differentiation engine that powers neural network training.

Usage in PyTorch
Let’s take a look at a single training step. 
For this example, we load a pretrained resnet18 model from torchvision. 
We create a random data tensor to represent a single image with 3 channels, and height & width of 64, 
and its corresponding label initialized to some random values. 
Label in pretrained models has shape (1,1000).
"""

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step() # gradient descent

