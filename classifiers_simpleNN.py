import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat

from torch.nn import Linear, Sequential, Module, Dropout, Conv1d
from torch.nn import CrossEntropyLoss, Embedding


def accuracy(out_class, y):
    '''
    Compute accuracy based on output and target classes.
    '''
    classes = torch.argmax(out_class, dim=-1)
    accuracy = torch.eq(classes, y)
    return accuracy


class ClassifierModule(Module):
    '''
    Implements a fully-connected neural network with variable number of layers.
    '''
    def __init__(self, args, input_dim):
        super(ClassifierModule, self).__init__()
        layers = []

        # first layer projects the input
        layers.append(Linear(input_dim, args.units[0]))

        # initialize a variable number of hidden layers based on args
        for i, u in enumerate(args.units[:-1]):
            layers.append(Linear(u, args.units[i+1]))

        # final layer projects to the output classes
        layers.append(Linear(args.units[-1], args.num_classes))

        self.layers = Sequential(*layers)
        self.dropout = Dropout(p=args.p_drop)
        self.activation = args.activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(self.dropout(layer(x)))

        return self.layers[-1](x)


class SimpleClassifier(Module):
    '''
    Implements a fully-connected neural network classifier.
    '''
    def __init__(self, args):
        super(SimpleClassifier, self).__init__()
        self.args = args
        self.losses = {'train': np.array([4]), 'val': np.array([4])}

        self.criterion_class_nored = CrossEntropyLoss(reduction='none').cuda()
        self.criterion_class = CrossEntropyLoss().cuda()

        self.build_model(args)

    def build_model(self, args):
        chn = args.num_channels

        # start with a dimension reduction over the channels
        self.spatial_conv = Conv1d(chn, args.dim_red, kernel_size=1, groups=1)
        self.classifier = ClassifierModule(args, args.dim_red*args.sample_rate)

    def loaded(self, args):
        self.args = args
        self.inputs = []
        self.targets = []

    def forward(self, x, sid=None):
        '''
        Run a dimension reduction over the channels then run the classifier.
        '''
        x = self.spatial_conv(x)
        x = self.classifier.activation(self.classifier.dropout(x))
        x = self.classifier(x.reshape(x.shape[0], -1))

        return None, x

    def end(self):
        pass

    def end_(self):
        # only used for diagnostic purposes
        for split in ['train', 'val']:
            ex = 2478 if split == 'train' else 590
            losses = self.losses[split]
            print(losses.shape)

            losses = losses[1:].reshape(-1, ex)
            path = os.path.join(
                self.args.result_dir, 'loss_hist_' + split, 'd')

            for i, loss in enumerate(losses[::int(losses.shape[0]/10)]):
                plt.hist(loss, bins=100)
                plt.savefig(path + str(i))
                plt.close('all')

    def end__(self):
        # only used for diagnostic purposes
        losses = self.losses['val'][-590:]

        inputs = np.array(self.inputs)
        inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3])
        targets = np.array(self.targets).reshape(-1)

        inds = list(np.argsort(losses))
        inputs = inputs[inds, :, :]
        targets = targets[inds]

        inps = np.concatenate((inputs[:10, :, :], inputs[-10:, :, :]))
        savemat(os.path.join(self.args.result_dir, 'inputs.mat'), {'X': inps})
        tars = np.concatenate((targets[:10], targets[-10:]))
        savemat(os.path.join(self.args.result_dir, 'targets.mat'), {'X': tars})

    def loss_reg(self):
        '''
        Apply regularization on the weights.
        '''
        new_weights = [layer.weight.view(-1) for layer in self.classifier.layers]
        new_weights.append(self.spatial_conv.weight.view(-1))

        new_weights = torch.cat(new_weights)
        return torch.linalg.norm(new_weights, ord=1)

    def loss(self, x, i=0, sid=None, train=True, criterion=None):
        '''
        Run the model in forward mode and compute loss for this batch.
        '''
        inputs = x[:, :self.args.num_channels, :]
        targets = x[:, -1, 0].long()
        out_pred, out_class = self.forward(inputs, sid)

        # compute loss for each sample
        loss = self.criterion_class_nored(out_class, targets)

        # for validation the top 40% losses are more informative
        if not train:
            loss = torch.quantile(loss, 0.4)

        loss = torch.mean(loss)

        # apply regularization if needed
        if self.args.l1_loss:
            loss += self.args.alpha_norm * self.loss_reg()

        # compute accuracy
        acc = accuracy(out_class, targets).float()
        if criterion is None:
            acc = torch.mean(acc)

        # assemble dictionary of losses
        losses = {'trainloss/optloss/Training loss: ': loss,
                  'trainloss/Train accuracy: ': acc,
                  'valloss/Validation loss: ': loss,
                  'valloss/valcriterion/Validation accuracy: ': acc,
                  'valloss/saveloss/none': 1-acc}

        return losses, out_pred, None

    def loss_(self, x, i=0, sid=None, train=True, criterion=None):
        # only used for diagnostic purposes
        tr = 'train' if train else 'val'
        loss = loss.detach().cpu().numpy()
        self.losses[tr] = np.concatenate((self.losses[tr], loss))
        self.inputs.append(inputs.cpu().numpy())
        self.targets.append(targets.cpu().numpy())


class SimpleClassifier0(SimpleClassifier):
    '''
    Simple Classifier but with a single linear transform.
    Used for testing.
    '''
    def build_model_(self, args):
        self.classifier = Linear(args.num_channels*args.sample_rate,
                                 args.num_classes)
        self.dropout = Dropout(p=args.p_drop)

    def forward(self, x, sid=None):
        x = self.classifier.dropout(x.reshape(x.shape[0], -1))
        x = self.classifier(x)

        return None, x


class SimpleClassifierSemb(SimpleClassifier):
    '''
    Simple Classifier for multi-subject decoding using subject embeddings.
    '''
    def __init__(self, args):
        super(SimpleClassifierSemb, self).__init__(args)

        # channel dimension is increased with the embedding dimension
        in_c = args.num_channels + args.embedding_dim
        self.spatial_conv = Conv1d(in_c, args.dim_red, kernel_size=1, groups=1)
        self.subject_emb = Embedding(args.subjects, args.embedding_dim)

    def forward(self, x, sid=None):
        # concatenate subject embeddings with input data
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)
        x = torch.cat((x, sid), dim=1)

        return super(SimpleClassifierSemb, self).forward(x)
