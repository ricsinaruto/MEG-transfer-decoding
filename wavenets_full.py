from torch.nn import Sequential, Conv1d, Module, CrossEntropyLoss, Embedding
from torch.nn import Softmax, Linear
import torch
import torch.nn.functional as F
import numpy as np

import os

from wavenets_simple import WavenetSimple
from cichy_data import mulaw_inv


def accuracy(out_class, y):
    '''
    Compute accuracy based on output and target classes.
    '''
    classes = torch.argmax(out_class, dim=-1)
    accuracy = torch.eq(classes, y)
    return accuracy, classes


def wave_init_weights(m):
    """Initialize conv1d with Xavier_uniform weight and 0 bias."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=1e-3, std=1e-2)


class WavenetLayer(Module):
    def __init__(self,
                 shift,
                 kernel_size,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 cond_channels=None,
                 in_channels=None,
                 bias=False):
        super(WavenetLayer, self).__init__()

        in_channels = in_channels or residual_channels
        self.shift = shift
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.cond_channels = cond_channels

        self.conv_dilation = Conv1d(
            in_channels,
            2 * dilation_channels,  # We stack W f,k and W g,k, similar to PixelCNN
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

        self.conv_res = Conv1d(
            dilation_channels,
            residual_channels,
            kernel_size=1,
            bias=bias,
        )
        self.conv_skip = Conv1d(
            dilation_channels,
            skip_channels,
            kernel_size=1,
            bias=bias,
        )

        self.conv_cond = None
        if cond_channels is not None:
            self.conv_cond = Conv1d(
                cond_channels,
                dilation_channels * 2,
                kernel_size=1,
                bias=bias,
            )

        self.conv_input = None
        if in_channels != residual_channels:
            self.conv_input = Conv1d(
                in_channels,
                residual_channels,
                kernel_size=1,
                bias=bias,
            )

    def forward(self, x, c, causal_pad=False):
        """Compute residual and skip output from inputs x.
        Args:
            x: (B,C,T) tensor where C is the number of residual channels
                when `in_channels` was specified the number of input channels
            c: optional tensor containing a global (B,C,1) or local (B,C,T)
                condition, where C is the number of condition channels.
            causal_pad: layer performs causal padding when set to True, otherwise
                assumes the input is already properly padded.
        Returns
            r: (B,C,T) tensor where C is the number of residual channels
            skip: (B,C,T) tensor where C is the number of skip channels
        """
        p = (self.causal_left_pad, 0) if causal_pad else (0, 0)
        x_dilated = self.conv_dilation(F.pad(x, p))

        if self.cond_channels:
            assert c is not None, "conditioning required"
            x_cond = self.conv_cond(c)
            x_dilated = x_dilated + x_cond[:, :, -x_dilated.shape[-1]:]
        x_filter = torch.tanh(x_dilated[:, :self.dilation_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.dilation_channels:])
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        res = self.conv_res(x_h)

        if self.conv_input is not None:
            x = self.conv_input(x)  # convert to res channels

        if causal_pad:
            out = x + res
        else:
            out = x[..., -res.shape[-1]:] + res

        # need to keep only second half of skips
        return out, skip[:, :, -self.shift:]


class WaveNetLogitsHead(Module):
    def __init__(
        self,
        skip_channels,
        residual_channels,
        head_channels,
        out_channels,
        bias=True,
    ):
        """Collates skip results and transforms them to logit predictions.
        Args:
            skip_channels: number of skip channels
            residual_channels: number of residual channels
            head_channels: number of hidden channels to compute result
            out_channels: number of output channels
            bias: When true, convolutions use a bias term.
        """
        del residual_channels
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.LeakyReLU(),  # note, we perform non-lin first (i.e on sum of skips) # noqa:E501
            torch.nn.Conv1d(
                skip_channels,
                head_channels,
                kernel_size=1,
                bias=bias,
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                head_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            ),  # logits
        )

    def forward(self, encoded, skips):
        """Compute logits from WaveNet layer results.
        Args:
            encoded: unused last residual output of last layer
            skips: list of skip connections of shape (B,C,T) where C is
                the number of skip channels.
        Returns:
            logits: (B,Q,T) tensor of logits, where Q is the number of output
            channels.
        """
        del encoded
        return self.transform(sum(skips))


class WavenetFull(WavenetSimple):
    '''
    The full wavenet model as described in the original paper
    '''
    def __init__(self, args):
        super(WavenetFull, self).__init__(args)
        self.criterion = CrossEntropyLoss(reduction='none').cuda()

    def loaded(self, args):
        super().loaded(args)
        self.losses = []

    def build_model(self, args):
        self.quant_levels = args.mu + 1
        shift = args.sample_rate - args.rf + 1

        # embeddings for various conditioning
        self.subject_emb = Embedding(args.subjects, args.embedding_dim)
        self.cond_emb = Embedding(args.num_classes, args.class_emb)
        self.quant_emb = Embedding(self.quant_levels, args.quant_emb)
        self.inv_qemb = Linear(args.quant_emb, self.quant_levels, bias=False)

        self.softmax = Softmax(dim=-1)

        self.pca_w = Conv1d(args.num_channels, args.dim_red, kernel_size=1)

        # initial convolution
        layers = [
            WavenetLayer(
                shift=shift,
                kernel_size=1,
                dilation=1,
                in_channels=args.dim_red*args.quant_emb,
                residual_channels=args.residual_channels,
                dilation_channels=args.dilation_channels,
                skip_channels=args.skip_channels,
                cond_channels=args.cond_channels,
                bias=True,
            )
        ]

        layers += [
            WavenetLayer(
                shift=shift,
                kernel_size=args.kernel_size,
                dilation=d,
                residual_channels=args.residual_channels,
                dilation_channels=args.dilation_channels,
                skip_channels=args.skip_channels,
                cond_channels=args.cond_channels,
                bias=args.conv_bias,
            )
            for d in args.dilations
        ]

        self.layers = torch.nn.ModuleList(layers)

        self.conditioning_channels = args.cond_channels
        self.out_channels = args.num_channels * args.quant_emb
        
        self.logits = WaveNetLogitsHead(
            skip_channels=args.skip_channels,
            residual_channels=args.residual_channels,
            head_channels=args.head_channels,
            out_channels=self.out_channels,
            bias=True,
        )

        self.apply(wave_init_weights)

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

        # cond: B x E x T
        cond_ind = data['condition']
        cond = self.cond_emb(cond_ind.squeeze()).permute(0, 2, 1)

        # set elements of cond to 0 where cond_ind is 0
        cond = cond * (cond_ind > 0).float()

        # apply embedding to inputs and squeeze embeddings to last dim
        x = self.quant_emb(x)
        timesteps = x.shape[-2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        
        # apply pca and expand last dim, then reshape to (B, C, T)
        x = self.pca_w(x)
        x = x.reshape(x.shape[0], x.shape[1], timesteps, -1)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x.shape[0], -1, timesteps)

        skips = []
        for layer in self.layers:
            x, skip = layer(x, c=cond, causal_pad=causal_pad)
            skips.append(skip)

        out = self.logits(x, skips)

        # reshape to get (B, C, Q, T) -> (B, C, T, Q)
        out = out.reshape(
            out.shape[0], self.args.num_channels, -1, out.shape[-1])
        out = out.permute(0, 1, 3, 2)

        # apply transposed embedding to outputs
        #out = out @ self.quant_emb.weight.T
        out = self.inv_qemb(out)

        return out

    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        logits = self.forward(data)

        # have to make sure this exactly matches the inteded targets
        targets = data['targets']
        targets = targets[:, :, -logits.shape[-2]:]

        shape = targets.shape
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])

        loss = self.criterion(logits, targets)
        loss = torch.mean(loss)

        acc, preds = accuracy(logits, targets)
        acc = torch.mean(acc.float())

        '''
        pred_cont = mulaw_inv(preds)
        target_cont = mulaw_inv(targets)

        # compute MSE
        mse = self.mse_loss(pred_cont, target_cont)
        '''
        mse = acc

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss,
                  'valloss/Validation accuracy: ': acc,
                  'trainloss/Train accuracy: ': acc,
                  'trainloss/Training MSE: ': mse,
                  'valloss/Validation MSE: ': mse}

        #return losses, pred_cont.reshape(shape), target_cont.reshape(shape)
        return losses, None, None

    def compute_log_pxy(self, data, tau=1.0):
        '''
        Computes p(X=x,Y=y) and returns a BxQ tensor.
        To compute p(X,Y) from p(X|Y) we need to make an assumption about p(Y).
        We assume a uniform distribution over 118 classes.
        What about data without stimuli?
        Args:
            data: (B,Q,T) tensor of MEG data
            tau: temperature scaling parameter.
        Returns:
            log_pxy: (B,Q) tensor, where Q is the number of quantization levels

        log p(X=x) = log sum_y p(X=x|Y=y)p(Y=y)
                   = log sum_y exp(log p(X=x|Y=y)p(Y=y))
                   = log sum_y exp(log p(X=x|Y=y) + log p(Y=y))
                   = log sum_y exp(log p(X0|Y=y)p(X1|X0,Y=y)...p(XT|XT-1...X0,Y=y) + log p(Y=y))
                   = log sum_y exp(log p(X0|Y=y)+log p(X1|X0,Y=y)+...+log p(XT|XT-1...X0,Y=y) + log p(Y=y))
        this can be conveniently computed using torch.logsumexp
        '''
        Y = self.cond_channels
        Q = self.quantization_levels
        B, T = data.shape
        log_py = torch.log(torch.tensor(1.0 / 10))

        # All possible one-hot conditions
        y_conds = (
            (F.one_hot(torch.arange(0, Y, 1), num_classes=Y).permute(0, 1).view(Y, Y, 1))
            .to(data.device)
            .float()
        )

        log_pxys = []
        # For each class
        for y in range(10):
            # First, compute p(X|Y=y)
            logits, _ = self.forward(x=data, c=y_conds[y : y + 1])  # (B,Q,T)
            log_px_y = F.log_softmax(logits / tau, 1)
            # Note, wavenet predictions for the i-th sample is the distribution p(X_(i+1)|X_<=i, Y=y).
            # To compute get the value p(X=x|Y=y) we need to shift the image by one sample. Effectively,
            # we are not including p(X0).
            log_px_y = log_px_y[..., :-1]
            log_px_y = log_px_y.permute(0, 2, 1).reshape(-1, Q)  # (B*T,Q)
            log_px_y = log_px_y[torch.arange(B * (T - 1)), data[:, 1:].reshape(-1)].view(
                B, (T - 1)
            )  # (B,T)
            log_pxy = log_px_y[:, :data.shape[1]].sum(-1) + log_py
            log_pxys.append(log_pxy.clone())
            del logits
        return torch.stack(log_pxys, -1)

    def classify(self, data):
        log_pxys = self.compute_log_pxy(data)
        log_px = torch.logsumexp(log_pxys, -1)
        py_x = torch.exp(log_pxys - log_px.unsqueeze(-1))
        return py_x, data['targets']


class WavenetFullEmbPca(WavenetFull):
    '''
    Same as WavenetFull, except the pca_w is not shared across the
    embedding dimension.
    '''
    def build_model(self, args):
        super().build_model(args)

        in_channels = args.num_channels * args.quant_emb
        out_channels = args.dim_red * args.quant_emb
        self.pca_w = Conv1d(
            in_channels, out_channels, kernel_size=1, groups=args.quant_emb)

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

        # cond: B x E x T
        cond_ind = data['condition']
        cond = self.cond_emb(cond_ind.squeeze()).permute(0, 2, 1)

        # set elements of cond to 0 where cond_ind is 0
        cond = cond * (cond_ind > 0).float()

        # apply embedding to inputs and squeeze embeddings to last dim
        x = self.quant_emb(x)
        timesteps = x.shape[-2]
        x = x.permute(0, 3, 1, 2)  # B x E x C x T
        x = x.reshape(x.shape[0], -1, timesteps)  # B x (C*E) x T
        
        # apply pca separately along embedding dimension
        x = self.pca_w(x)

        skips = []
        for layer in self.layers:
            x, skip = layer(x, c=cond, causal_pad=causal_pad)
            skips.append(skip)

        out = self.logits(x, skips)

        # reshape to get (B, C, Q, T) -> (B, C, T, Q)
        out = out.reshape(
            out.shape[0], self.args.num_channels, -1, out.shape[-1])
        out = out.permute(0, 1, 3, 2)

        # apply transposed embedding to outputs
        out = self.inv_qemb(out)

        return out


class WavenetFullSimple(WavenetSimple):
    '''
    Simple version of wavenet with residual and skip connections.
    '''
    def build_model(self, args):
        super(WavenetFullSimple, self).build_model(args)

        del self.last_conv
        # add two 1x1 convolutions at the end
        self.last_conv = [
            Conv1d(self.ch, self.ch, kernel_size=1, groups=args.groups),
            Conv1d(self.ch, args.num_channels, kernel_size=1)]

        self.last_conv = Sequential(*self.last_conv)

    def forward(self, x, sid):
        x = self.first_conv(x)
        skips = []

        for conv in self.cnn_layers:
            x = self.dropout(x)
            xf = conv(x)
            xf = self.activation(xf)

            x = x[:, :, -xf.shape[2]:] + xf
            skips.append(x)

        # sum together skip connections
        length = skips[-1].shape[2]
        x = skips[0][:, :, -length:]
        for xs in skips[1:]:
            x = x + xs[:, :, -length:]

        x = self.activation(x)
        x = self.last_conv[0](x)
        out = self.activation(x)
        out = self.last_conv[1](out)

        return out, x

    def residual(self, data, data_f):
        '''
        This function is needed to make kernel_network_FIR compatible
        with this model. It simply applies the residual connection.
        '''
        return data[:, :, -data_f.shape[2]:] + data_f


class WavenetSimpleSkips(WavenetFullSimple):
    '''
    Simple version of wavenet with residual and skip connections.
    '''
    def build_model(self, args):
        super(WavenetSimpleSkips, self).build_model(args)
        self.conv_skip = Sequential(*[
            Conv1d(self.ch, self.ch, kernel_size=1) for _ in self.cnn_layers])

    def forward(self, x, sid=None):
        x = self.first_conv(x)
        skips = []

        for conv, conv_skip in zip(self.cnn_layers, self.conv_skip):
            xo = self.activation(self.dropout(conv(x)))
            xo = conv_skip(xo)

            x = x[:, :, -xo.shape[2]:] + xo
            skips.append(xo)

        # sum together skip connections
        length = skips[-1].shape[2]
        x = skips[0][:, :, -length:]
        for xs in skips[1:]:
            x = x + xs[:, :, -length:]

        return None, x
