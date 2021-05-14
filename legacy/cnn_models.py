from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module
from torch.nn import Dropout, Dropout2d, Conv1d, MaxPool1d
from torchvision import transforms


class SimpleCNN(Module):
    def __init__(self,
                 mean,
                 std,
                 channels,
                 dropout,
                 final_channels,
                 num_classes):
        super(SimpleCNN, self).__init__()

        self.norm = transforms.Normalize(mean, std)
        self.dropout = dropout
        self.ch = channels * 2

        self.cnn_layers = Sequential(

            Conv2d(self.ch, self.ch, kernel_size=5, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(self.ch, self.ch, kernel_size=1, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(p=self.dropout),

            Conv2d(self.ch, final_channels, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(final_channels, final_channels, kernel_size=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(p=self.dropout),

            Conv2d(final_channels, final_channels, kernel_size=1),
            ReLU(inplace=True),

        )

        self.linear_layers = Sequential(
            Dropout(p=self.dropout),
            Linear(final_channels * 33, num_classes)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size(1))
        x = self.linear_layers(x)
        return x


class Simple1DCNN(Module):
    mean = 0
    std = 0
    channels = 0
    dropout = 0
    final_channels = 0
    num_classes = 0

    def __init__(self):
        super(Simple1DCNN, self).__init__()

        self.dropout = Simple1DCNN.dropout
        self.ch = Simple1DCNN.channels
        self.final_ch = Simple1DCNN.final_channels

        self.norm = transforms.Normalize(Simple1DCNN.mean, Simple1DCNN.std)

        self.cnn_layers = Sequential(
            Conv1d(self.ch, self.ch+20, kernel_size=7, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            Dropout(p=self.dropout),

            Conv1d(self.ch+20, self.ch+60, kernel_size=5, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            Dropout(p=self.dropout),

            Conv1d(self.ch+60, self.ch+120, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            Dropout(p=self.dropout),

            Conv1d(self.ch+120, self.ch+200, kernel_size=1, padding=0),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),

        )

        self.linear_layers = Sequential(
            Dropout(self.dropout),
            Linear((self.ch+200) * 10, Simple1DCNN.num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2), 1)
        x = self.norm(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size(1))
        x = self.linear_layers(x)
        return x
