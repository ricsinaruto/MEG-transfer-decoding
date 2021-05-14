import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Sigmoid, Dropout2d
from torch.optim import Adam, SGD
from torchvision import transforms


class SimpleFFN(Module):
    features = 0
    classes = 0
    dropout = 0

    def __init__(self):
        super(SimpleFFN, self).__init__()

        self.linear_layers = Sequential(
            Linear(SimpleFFN.features, 500),
            ReLU(),
            Dropout(p=SimpleFFN.dropout),
            Linear(500, SimpleFFN.classes),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.linear_layers(x)
        return x