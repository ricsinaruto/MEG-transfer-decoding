import numpy as np
import os

import torch
from torch.nn import Linear, Sequential, Module, Dropout, Dropout2d, SELU, ReLU, Conv1d
from torch.nn import MSELoss, CrossEntropyLoss
import matplotlib.pyplot as plt

from wavenets_simple import WavenetSimple, WavenetSimpleSTS, WavenetSimpleChannelUp


def accuracy(out_class, y):
    classes = torch.argmax(out_class, dim=1)
    accuracy = torch.sum(torch.eq(classes, y))/y.shape[0]
    return accuracy


class ClassifierModule(Module):
    def __init__(self, args, input_dim):
        super(ClassifierModule, self).__init__()

        self.l1 = Linear(input_dim, 800)
        self.l2 = Linear(800, 400)
        self.l3 = Linear(400, args.num_classes)
        self.dropout = Dropout(p=args.p_drop)
        self.activation = args.activation

    def forward(self, x):
        x = self.activation(self.dropout(self.l1(x)))
        x = self.activation(self.dropout(self.l2(x)))
        x = self.l3(x)

        return x


class SimpleClassifier(Module):
    '''
    This class adds a classifier on top of the normal wavenet.
    '''
    def __init__(self, args):
        super(SimpleClassifier, self).__init__()
        self.args = args

        self.classifier = ClassifierModule(args, 80*args.rf)
        self.losses = {'train': np.array([4]), 'val': np.array([4])}

        self.spatial_conv = Conv1d(args.num_channels, 80, kernel_size=1, groups=1)

        self.criterion_class_nored = CrossEntropyLoss(reduction='none').cuda()
        self.criterion_class = CrossEntropyLoss().cuda()

    def loaded(self, args):
        self.args = args

    def forward(self, x):
        x = self.classifier.activation(self.classifier.dropout(self.spatial_conv(x)))
        x = x.reshape(x.shape[0], -1)

        x = self.classifier(x)

        return None, x

    def end(self):
        for split in ['train', 'val']:
            ex = 2478 if split == 'train' else 590
            losses = self.losses[split]
            print(losses.shape)

            losses = losses[1:].reshape(-1, ex)
            path = os.path.join(self.args.result_dir, 'loss_hist_' + split, 'd')

            for i, loss in enumerate(losses[::int(losses.shape[0]/10)]):
                plt.hist(loss, bins=100)
                plt.savefig(path + str(i))
                plt.close('all')

    def loss(self, x, i=0, sid=None, train=True):
        '''
        Run the model in forward mode and compute loss for this batch.
        '''
        inputs = x[:, :self.args.num_channels, :]
        targets = x[:, -1, 0].long()
        out_pred, out_class = self.forward(inputs)

        loss = self.criterion_class_nored(out_class, targets)
        tr = 'train' if train else 'val'
        self.losses[tr] = np.concatenate((self.losses[tr], loss.detach().cpu().numpy()))

        #loss_class = self.criterion_class(out_class, targets)
        if not train:
            loss = torch.quantile(loss, 0.4)
        loss = torch.mean(loss)

        return loss, out_pred, None, accuracy(out_class, targets).float()


class WavenetClassifier(SimpleClassifier):
    '''
    This class adds a classifier on top of the normal wavenet.
    '''
    def __init__(self, args):
        super(WavenetClassifier, self).__init__(args)
        self.wavenet = WavenetSimple(args)
        self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)

        # might need separate neural network for each timestep
        self.classifier = ClassifierModule(args, self.class_dim)

        # separate losses for next timestep prediciton and classification
        self.criterion_pred = MSELoss().cuda()

    def forward(self, x):
        # for classification we don't need to do the last 1x1 convolution
        output, x = self.wavenet(x)
        x = x[:, :, ::self.args.rf].reshape(x.shape[0], -1)
        x = self.classifier(x)

        return output, x


class WavenetClassPred(WavenetClassifier):
    '''
    This class adds a classifier on top of the normal wavenet.
    '''

    def forward(self, x):
        # for classification we don't need to do the last 1x1 convolution
        output, x = self.wavenet.forward3(x)
        x = x[:, :, ::self.args.rf].reshape(x.shape[0], -1)
        x = self.classifier(x)

        return output, x

    def loaded(self, args):
        super(WavenetClassPred, self).loaded(args)
        self.classifier = ClassifierModule(args, self.class_dim)
        self.wavenet.dropout = Dropout2d(args.p_drop)

        #for parameter in self.wavenet.parameters():
        #    parameter.requires_grad = False

    def loss(self, x, i=0, sid=None, train=True):
        '''
        Run the model in forward mode and compute loss for this batch.
        '''
        accuracy = torch.zeros([1])
        #x = x[:, :, :-self.args.timesteps]
        if self.args.pred:
            loss, _, _, _ = self.wavenet.loss(x, train)
        else:
            loss, out_pred, _, accuracy = super(WavenetClassPred, self).loss(x, train=train)

            #target = x[:, :self.args.num_channels, -out_pred.shape[2]:]
            #loss_pred = self.criterion_pred(out_pred, target)

        return loss, None, None, accuracy
