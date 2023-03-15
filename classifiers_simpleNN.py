import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat

from torch.nn import Linear, Sequential, Module, Dropout, Conv1d
from torch.nn import CrossEntropyLoss, Embedding

from wavenets_full import WavenetFull
from cichy_data import mulaw_inv


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

        if args.trial_average:
            args.sample_rate = args.trial_average[1] - args.trial_average[0]

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

        if self.args.trial_average:
            timing = self.args.trial_average
            x = x.reshape(x.shape[0], x.shape[1], 4, -1)
            x = torch.mean(x, dim=2)
            x = x[:, :, timing[0]:timing[1]]

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
        new_weights = [layr.weight.view(-1) for layr in self.classifier.layers]
        new_weights.append(self.spatial_conv.weight.view(-1))

        new_weights = torch.cat(new_weights)
        return torch.linalg.norm(new_weights, ord=1)

    def gradient_analysis(self, args):
        '''
        1. Create a batch of random inputs.
        2. Compute the gradient of the loss with respect to the input.
        '''
        grads = []
        for i in range(100):
            args.num_channels = 306
            num_samples = 100
            self.eval()
            self.criterion_class_nored = CrossEntropyLoss(reduction='none').cuda()

            # create a batch of random inputs
            x = torch.randn(
                (num_samples, args.num_channels, args.sample_rate),
                requires_grad=True,
                device='cuda')

            # add target classes to the 2nd dimension of x
            y = torch.randint(0, args.num_classes, (num_samples,)).cuda()
            x = torch.cat((x, y.reshape(-1, 1, 1).repeat(1, 1, x.shape[2])),
                        dim=1)

            x.retain_grad()

            # create sid
            sid = torch.randint(0, args.subjects, (num_samples,)).cuda()

            # compute the gradient of the loss with respect to the input
            losses, _, _ = self.loss(x, sid=sid)
            losses['trainloss/optloss/Training loss: '].backward()

            grad = x.grad[:, :args.num_channels, :]
            grad = grad.detach().cpu().numpy()
            grads.append(grad)

        # save the gradient
        grads = np.concatenate(grads)
        np.save(os.path.join(args.result_dir, 'grads.npy'), grads)

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

        out_class = torch.argmax(out_class, dim=-1)
        return losses, (out_class, out_pred), targets


class SimpleClassPred(WavenetFull):
    def build_model(self, args):
        self.quant_levels = args.mu + 1
        inp_ch = args.num_channels * args.quant_emb
        out_ch = args.num_channels * args.quant_emb
        
        self.conv = Conv1d(inp_ch,
                           out_ch,
                           kernel_size=args.rf,
                           bias=False,
                           groups=args.num_channels)

        self.activation = args.activation

        self.quant_emb = Embedding(self.quant_levels, args.quant_emb)
        self.inv_qemb = Linear(args.quant_emb, self.quant_levels, bias=False)

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, data, causal_pad=False):
        """Computes logits and encoding results from observations.
        Args:
            x: (B,T) or (B,Q,T) tensor containing observations
            c: optional conditioning Tensor. (B,C,1) for global conditions,
                (B,C,T) for local conditions. None if unused
            causal_pad: Whether or not to perform causal padding.
        Returns:
            logits: (B,Q,T) tensor of logits. Note that the t-th temporal output
                represents the distribution over t+1.
            encoded: same as `.encode`.
        """
        x = data['inputs']

        '''
        Initially train without condition to be able to compare later.
        '''
        # cond: B x E x T
        #cond_ind = data['condition']
        #cond = self.cond_emb(cond_ind.squeeze()).permute(0, 2, 1)

        # set elements of cond to 0 where cond_ind is 0
        #cond = cond * (cond_ind > 0).float()

        # concatenate cond to x
        #x = torch.cat((x, cond), dim=1)

        # apply quantization embedding to x
        x = x @ self.quant_emb.weight
        timesteps = x.shape[-2]

        # B x C*Q x T
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x.shape[0], -1, timesteps)

        x = self.conv(x)
        x = self.activation(x)

        # B x C x T x Q
        x = x.reshape(x.shape[0], -1, self.args.quant_emb, x.shape[-1])
        x = x.permute(0, 1, 3, 2)

        x = self.inv_qemb(x)

        return x

    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        losses, pred_cont, target_cont = super().loss(data, i, sid, train, criterion)

        '''
        if i == 0 and train:
            pred_cont = pred_cont.detach().cpu().numpy()
            target_cont = target_cont.detach().cpu().numpy()

            # save predictions and targets
            path = os.path.join(self.args.result_dir, 'preds.npy')
            np.save(path, pred_cont)
            path = os.path.join(self.args.result_dir, 'targets.npy')
            np.save(path, target_cont)
        '''
        '''
        if i == 0 and train:
            inputs = data['inputs'].detach().cpu().numpy()
            targets = data['targets'].detach().cpu().numpy()

            # save predictions and targets
            path = os.path.join(self.args.result_dir, 'inputs.npy')
            np.save(path, inputs)
            path = os.path.join(self.args.result_dir, 'targets_full.npy')
            np.save(path, targets)
        '''

        return losses, None, None


class SimpleClassFakeLoss(SimpleClassPred):
    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        data['inputs'] = torch.ones_like(data['inputs'],
                                         dtype=torch.float32,
                                         requires_grad=True)
        # expand inputs with an extra dimension of size quant_levels
        data['inputs'] = data['inputs'].unsqueeze(-1)
        data['inputs'] = data['inputs'].repeat(1, 1, 1, self.quant_levels)
        data['inputs'].retain_grad()
        logits = self.forward(data)

        # have to make sure this exactly matches the inteded targets
        targets = data['targets']
        targets = targets[:, :, -logits.shape[-2]:]

        shape = targets.shape
        targets = targets.reshape(-1)

        loss = torch.sum(logits[0])

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss}

        return losses, None, None


class SimpleClassAutoregcheck(SimpleClassPred):
    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        data['inputs'] = torch.ones_like(data['inputs'],
                                         dtype=torch.float32,
                                         requires_grad=True)
        # expand inputs with an extra dimension of size quant_levels
        data['inputs'] = data['inputs'].unsqueeze(-1)
        data['inputs'] = data['inputs'].repeat(1, 1, 1, self.quant_levels)
        data['inputs'].retain_grad()
        logits = self.forward(data)

        # have to make sure this exactly matches the inteded targets
        targets = data['targets']
        targets = targets[:, :, -logits.shape[-2]:]
        logits = logits[:, :, -1, :]
        targets = targets[:, :, -1]
        

        shape = targets.shape
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])

        loss = self.criterion(logits, targets)
        loss = torch.mean(loss)

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss}

        return losses, None, None


class SimpleClassifierPosEncoding(SimpleClassifier):
    def __init__(self, args):
        super(SimpleClassifierPosEncoding, self).__init__(args)

        # initialize position look up table
        d = args.pos_enc_d
        vectors = []
        for t in range(1, 1000):
            k = np.arange(1, int(d/2) + 1)
            w = 1/10000**(2*k/d)

            a = np.sin(w*t)
            b = np.cos(w*t)

            p = np.empty((a.size + b.size,), dtype=a.dtype)
            p[0::2] = a
            p[1::2] = b

            p = torch.Tensor(p).float().cuda()
            vectors.append(p)

        self.vectors = torch.stack(vectors)

    def build_model(self, args):
        chn = args.num_channels + args.pos_enc_d - 1

        # start with a dimension reduction over the channels
        self.spatial_conv = Conv1d(chn, args.dim_red, kernel_size=1, groups=1)
        self.classifier = ClassifierModule(args, args.dim_red*args.sample_rate)

    def embed(self, x):
        encoding = self.vectors[x[:, -1, :].long()]
        encoding = encoding.permute(0, 2, 1)

        if self.args.pos_enc_type == 'cat':
            x = torch.cat((x[:, :-1, :], encoding), axis=1)
        else:
            x = x[:, :-1, :] + encoding

        return x

    def forward(self, x, sid=None):
        '''
        Run a dimension reduction over the channels then run the classifier.
        '''
        x = self.embed(x)
        return super(SimpleClassifierPosEncoding, self).forward(x, sid)


class SimpleClassifierTimeEncoding(SimpleClassifierPosEncoding):
    def __init__(self, args):
        super(SimpleClassifierTimeEncoding, self).__init__(args)

        self.inds = np.arange(self.args.sample_rate).reshape(1, -1)

    def build_model(self, args):
        chn = args.num_channels + args.pos_enc_d

        # start with a dimension reduction over the channels
        self.spatial_conv = Conv1d(chn, args.dim_red, kernel_size=1, groups=1)
        self.classifier = ClassifierModule(args, args.dim_red*args.sample_rate)

    def embed(self, x):
        inds = np.repeat(self.inds, x.shape[0], axis=0)
        encoding = self.vectors[torch.Tensor(inds).long()]
        encoding = encoding.permute(0, 2, 1)

        if self.args.pos_enc_type == 'cat':
            x = torch.cat((x, encoding), axis=1)
        else:
            x = x + encoding

        return x

    def forward(self, x, sid=None):
        '''
        Run a dimension reduction over the channels then run the classifier.
        '''
        x = self.embed(x)
        return super(SimpleClassifierPosEncoding, self).forward(x, sid)


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
