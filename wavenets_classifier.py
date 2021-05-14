import numpy as np

from torch.nn import Linear, Sequential, Module, Dropout, SELU
from torch.nn import MSELoss, CrossEntropyLoss

from wavenets_simple import Wavenet


class WavenetClassifier(Module):
    '''
    This class adds a classifier on top of the normal wavenet.
    '''
    def __init__(self, args):
        super(WavenetClassifier, self).__init__()
        self.ch = args.ch_mult * args.num_channels
        self.wavenet = Wavenet(args)

        # might need separate neural network for each timestep
        self.classifier = Sequential(
            Dropout(p=0.9),
            Linear(self.ch, args.num_classes),
            SELU(),
            Dropout(p=0.4),
            Linear(args.num_classes, args.num_classes)
        )

        # separate losses for next timestep prediciton and classification
        self.criterion_pred = MSELoss().cuda()
        self.criterion_class = CrossEntropyLoss().cuda()

    def forward(self, x):
        # for classification we don't need to do the last 1x1 convolution
        output, x = self.wavenet(x)
        x = x.permute(0, 2, 1)[:, -1, :]
        x = self.classifier(x)

        return output, x

    def loss(self, batch):
        '''
        Run the model in forward mode and compute loss for this batch.
        '''
        out_pred, out_class = self.forward(batch['x'][:, :, :-1])
        loss_pred = self.criterion_pred(out_pred, batch['x'][:, :, 1:])
        loss_class = self.criterion_class(out_class, batch['y'][:, 0])
        loss = loss_class + loss_pred
        return loss

    def accuracy(self, out_class, y):
        '''
        Compute accuracy for model predicted out_class and target y.
        '''
        classes = np.argmax(out_class.detach().cpu().numpy(), axis=1)
        accuracy = np.sum(np.equal(classes, y[:, 0].cpu().numpy()))/y.shape[0]
        return accuracy
