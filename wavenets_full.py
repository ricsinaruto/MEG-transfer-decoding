from torch.nn import Sequential, Conv1d, Module, CrossEntropyLoss, Embedding
from torch.nn import Softmax, Linear
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

# import savemat
from scipy.io import savemat

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
                 dropout=0.0,
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

        self.dropout = torch.nn.Dropout1d(p=dropout)

    def loaded(self, dropout=0.0, shift=None):
        self.dropout = torch.nn.Dropout1d(p=dropout)

        if shift:
            self.shift = shift

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
            x_cond = self.conv_cond(c[:, :, -x_dilated.shape[-1]:])
            x_dilated = x_dilated + x_cond
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

        # dropout
        out = self.dropout(out)

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
        dropout=0.0
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
            torch.nn.Dropout1d(p=dropout),
            torch.nn.LeakyReLU(),  # note, we perform non-lin first (i.e on sum of skips) # noqa:E501
            torch.nn.Conv1d(
                skip_channels,
                head_channels,
                kernel_size=1,
                bias=bias,
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.Dropout1d(p=dropout),
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
        self.mse_loss = torch.nn.MSELoss().cuda()

    def loaded(self, args):
        super().loaded(args)
        self.losses = []

        for layer in self.layers:
            layer.loaded(shift=args.skips_shift)

    def inv_qemb(self, x):
        return self.inv_qemb_l(x)

    def build_model(self, args):
        self.quant_levels = args.mu + 1
        shift = args.sample_rate - args.rf + 1

        # embeddings for various conditioning
        self.subject_emb = Embedding(args.subjects, args.embedding_dim)
        self.cond_emb = Embedding(args.num_classes, args.class_emb)
        self.quant_emb = Embedding(self.quant_levels, args.quant_emb)
        self.inv_qemb_l = Linear(args.quant_emb, self.quant_levels, bias=False)

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
                dropout=args.p_drop
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
                dropout=args.p_drop
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
            dropout=args.p_drop
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

        all_loss = self.criterion(logits, targets)
        loss = torch.mean(all_loss)

        acc, preds = accuracy(logits, targets)
        acc = torch.mean(acc.float())

        pred_cont = mulaw_inv(preds)
        target_cont = mulaw_inv(targets)

        # compute MSE
        mse = self.mse_loss(pred_cont, target_cont)

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss,
                  'valloss/Validation accuracy: ': acc,
                  'trainloss/Train accuracy: ': acc,
                  'trainloss/Training MSE: ': mse,
                  'valloss/Validation MSE: ': mse}

        if self.save_preds and False:
            

            pred_cont = pred_cont.detach().cpu().numpy()
            target_cont = target_cont.detach().cpu().numpy()
            all_loss = all_loss.detach().cpu().numpy()

            pred_cont = pred_cont.reshape(shape)
            target_cont = target_cont.reshape(shape)
            all_loss = all_loss.reshape(shape)

            targets = targets.reshape(shape)
            acc = torch.eq(targets[:, :, 1:], targets[:, :, :-1])
            losses['valloss/repeat acc: '] = torch.mean(acc.float())

            train = 'train' if train else 'val'
            # save predictions and targets
            path = os.path.join(self.args.result_dir, train + f'preds{i}.npy')
            np.save(path, pred_cont)
            path = os.path.join(self.args.result_dir, train + f'targets{i}.npy')
            np.save(path, target_cont)
            path = os.path.join(self.args.result_dir, train + f'losses{i}.npy')
            np.save(path, all_loss)

            # save data['sid']
            path = os.path.join(self.args.result_dir, train + f'sid{i}.npy')
            np.save(path, data['sid'].cpu().numpy())

        

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
        nc = self.args.num_classes
        Q = self.quant_levels
        B, C, T = data['inputs'].shape
        log_py = torch.log(torch.tensor(1.0 / nc))

        data['condition'] = data['condition'].squeeze(dim=1)
        cond_inds = data['condition'] > 0
        cond_inds[:, :self.args.rf] = False

        logits_ts = self.args.sample_rate - self.args.rf + 1
        horizon = self.args.sr_data
        targets = data['targets'][:, :, -logits_ts:]
        targets = targets[:, :, :horizon].reshape(-1)

        log_pxys = []
        # For each class
        for y in range(1, nc):
            # replace condition with current label
            data['condition'] = data['condition'].squeeze(dim=1)
            data['condition'][cond_inds] = y
            data['condition'] = data['condition'].unsqueeze(1)

            # First, compute p(X|Y=y)
            logits = self.forward(data)
            logits = logits[:, :, :horizon, :]

            T = logits.shape[2]
            log_px_y = F.log_softmax(logits / tau, -1)
            log_px_y = log_px_y.reshape(-1, Q)
            log_px_y = log_px_y[torch.arange(log_px_y.shape[0]), targets]
            log_px_y = log_px_y.view(B, C*T)

            log_pxy = log_px_y.sum(-1) + log_py
            log_pxys.append(log_pxy.detach())
            del logits

        return torch.stack(log_pxys, -1)

    def classify(self, data):
        baseline = self.args.sr_data // 10 + 10
        targets = data['condition'][:, 0, self.args.rf+baseline].clone()

        log_pxys = self.compute_log_pxy(data)
        log_px = torch.logsumexp(log_pxys, -1)
        log_py_x = log_pxys - log_px.unsqueeze(-1)

        acc = log_py_x.argmax(1) == targets

        return acc


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


class WavenetFullTest(WavenetFullEmbPca):
    '''
    Same as WavenetFull, except the pca_w is not shared across the
    embedding dimension.
    '''
    def build_model(self, args):
        super().build_model(args)

        self.subject_emb = None
        self.save_preds = False

    def loaded(self, args):
        super().loaded(args)

        self.save_preds = True

    def get_cond(self, data):
        cond = None
        if self.args.cond_channels > 0:
            # cond: B x E x T
            cond_ind = data['condition']
            cond = self.cond_emb(cond_ind.squeeze()).permute(0, 2, 1)

            # set elements of cond to 0 where cond_ind is 0
            cond = cond * (cond_ind > 0).float()

        return cond

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

        cond = self.get_cond(data)

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

        if x.shape[-1] != skip.shape[-1]:
            print(x.shape)
            print(skip.shape)

        out = self.logits(x, skips)

        # reshape to get (B, C, Q, T) -> (B, C, T, Q)
        out = out.reshape(
            out.shape[0], -1, self.args.num_channels, out.shape[-1])
        out = out.permute(0, 2, 3, 1)

        # apply transposed embedding to outputs
        out = self.inv_qemb(out)

        return out


class WavenetFullTransposeEmb(WavenetFullTest):
    def inv_qemb(self, x):
        out = x @ self.quant_emb.weight.T
        return (out, x)

    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        logits = self.forward(data)
        embs = logits[1]
        logits = logits[0]

        # have to make sure this exactly matches the inteded targets
        targets = data['targets']
        targets = targets[:, :, -logits.shape[-2]:]

        shape = targets.shape
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])

        all_loss = self.criterion(logits, targets)
        loss = torch.mean(all_loss)

        acc, preds = accuracy(logits, targets)
        acc = torch.mean(acc.float())

        pred_cont = mulaw_inv(preds)
        target_cont = mulaw_inv(targets)

        # compute MSE
        mse = self.mse_loss(pred_cont, target_cont)

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss,
                  'valloss/Validation accuracy: ': acc,
                  'trainloss/Train accuracy: ': acc,
                  'trainloss/Training MSE: ': mse,
                  'valloss/Validation MSE: ': mse}

        return losses, None, None


class WavenetFullTestSemb(WavenetFullTest):
    def build_model(self, args):
        super().build_model(args)
        self.set_covs(args)

    def set_covs(self, args):
        covs = []
        for i in range(1, 15):
            path = os.path.join(args.data_path,
                                f'subj{i}',
                                f'subj{i}_cov.npy')
            covs.append(torch.tensor(np.load(path)))

        self.covs = torch.stack(covs).float().cuda()
        self.covs = self.covs[:, ::10]

        # linear layer going from covariance to lower dim embedding
        self.subject_emb = torch.nn.Linear(self.covs.shape[1],
                                           args.embedding_dim)

    def get_cond(self, data):
        cond = None
        if self.args.cond_channels > 0:
            # cond: B x E x T
            cond_ind = data['condition']
            cond = self.cond_emb(cond_ind.squeeze()).permute(0, 2, 1)

            # set elements of cond to 0 where cond_ind is 0
            cond = cond * (cond_ind > 0).float()

        embs = None
        if self.args.embedding_dim > 0:
            # get subject-specific covariance
            covs = self.covs[data['sid'].reshape(-1)]
            covs = covs.reshape(x.shape[0], -1, covs.shape[1])

            # project to embedding space
            embs = self.subject_emb(covs)
            embs = embs.permute(0, 2, 1)

        if self.args.embedding_dim and self.args.cond_channels:
            # concatenate subject and condition embeddings
            cond = torch.cat([cond, embs], dim=1)

        return cond


class WavenetFullGauss(WavenetFullTest):
    def build_model(self, args):
        super().build_model(args)

        # create a tensor of one-hot gauss targets
        self.gauss_targets = gaussian_filter1d(np.eye(args.mu+1),
                                               sigma=10)
        self.gauss_targets = torch.Tensor(self.gauss_targets).cuda()

    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        logits = self.forward(data)

        # have to make sure this exactly matches the inteded targets
        targets = data['targets']
        targets = targets[:, :, -logits.shape[-1]:]
        targets = self.gauss_targets[targets]

        targets = targets.reshape(-1, logits.shape[-1])
        logits = logits.reshape(-1, logits.shape[-1])

        loss = self.criterion(logits, targets)
        loss = torch.mean(loss)

        targets = torch.argmax(targets, dim=-1)
        acc, preds = accuracy(logits, targets)
        acc = torch.mean(acc.float())

        pred_cont = mulaw_inv(preds)
        target_cont = mulaw_inv(targets)

        # compute MSE
        mse = self.mse_loss(pred_cont, target_cont)

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss,
                  'valloss/Validation accuracy: ': acc,
                  'trainloss/Train accuracy: ': acc,
                  'trainloss/Training MSE: ': mse,
                  'valloss/Validation MSE: ': mse}

        return losses, None, None


class WavenetFullChannel(WavenetFullTest):
    def build_model(self, args):
        self.quant_levels = args.mu + 1
        shift = args.sample_rate - args.rf + 1
        self.save_preds = False

        # embeddings for various conditioning
        self.cond_emb = Embedding(args.num_classes, args.class_emb)

        # 306 quantazation embedding layers
        self.quant_emb = torch.randn(
            size=(args.num_channels, self.quant_levels, args.quant_emb),
            dtype=torch.float32,
            requires_grad=True,
            device='cuda')
        self.quant_emb = torch.nn.Parameter(self.quant_emb)

        self.softmax = Softmax(dim=-1)

        # initial convolution
        layers = [
            WavenetLayer(
                shift=shift,
                kernel_size=1,
                dilation=1,
                in_channels=args.quant_emb,
                residual_channels=args.residual_channels,
                dilation_channels=args.dilation_channels,
                skip_channels=args.skip_channels,
                cond_channels=args.cond_channels,
                bias=True,
                dropout=args.p_drop
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
                dropout=args.p_drop
            )
            for d in args.dilations
        ]

        self.layers = torch.nn.ModuleList(layers)

        self.conditioning_channels = args.cond_channels
        
        self.logits = WaveNetLogitsHead(
            skip_channels=args.skip_channels,
            residual_channels=args.residual_channels,
            head_channels=args.head_channels,
            out_channels=self.quant_levels,
            bias=True,
            dropout=args.p_drop
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

        cond = None
        if self.args.cond_channels > 0:
            # cond: B x E x T
            cond_ind = data['condition']
            try:
                inds = torch.squeeze(cond_ind, dim=1)
                cond = self.cond_emb(inds).permute(0, 2, 1)
            except RuntimeError:
                print(cond_ind.shape)
                print(x.shape)
                print(inds.shape)
                raise

            # set elements of cond to 0 where cond_ind is 0
            cond = cond * (cond_ind > 0).float()

            # repeat cond across new args.channels dim
            cond = cond.unsqueeze(1).repeat(1, self.args.num_channels, 1, 1)
            cond = cond.reshape(-1, cond.shape[-2], cond.shape[-1])

        # apply embedding to each channel separately
        timesteps = x.shape[-1]
        x = x.permute(0, 2, 1)  # B x T x C
        x = x.reshape(-1, x.shape[-1])  # B*T x C
        x = self.quant_emb[torch.arange(x.shape[-1]), x]  # B*T x C x E

        # reshape back
        x = x.reshape(-1, timesteps, x.shape[-2], x.shape[-1])  # B x T x C x E
        x = x.permute(0, 2, 3, 1)  # B x C x E x T
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # B*C x E x T

        skips = []
        for layer in self.layers:
            x, skip = layer(x, c=cond, causal_pad=causal_pad)
            skips.append(skip)

        if x.shape[-1] != skip.shape[-1]:
            print(x.shape)
            print(skip.shape)

        out = self.logits(x, skips)

        # reshape to get (B*C, Q, T) -> (B, C, T, Q)
        out = out.reshape(
            -1, self.args.num_channels, out.shape[-2], out.shape[-1])
        out = out.permute(0, 1, 3, 2)

        return out

    def generate(self, train=None):
        '''
        Recursively generate with a trained model in various ways.
        '''
        self.eval()
        input_mode = self.args.generate_input
        mode = self.args.generate_mode
        sampling = self.args.generate_sampling
        noise = self.args.generate_noise
        channels = self.args.num_channels
        shift = self.args.rf
        gen_len = self.args.generate_length

        output = torch.zeros((channels, gen_len)).cuda()

        if input_mode == 'gaussian_noise':
            # input is gaussian noise
            data = torch.normal(0.0, noise, size=(channels, gen_len)).cuda()
        elif input_mode == 'none':
            data = torch.normal(0.0, noise, size=(channels, shift))
            data = torch.cat((data, torch.zeros((channels, gen_len))), dim=1)
            data = data.cuda()
        elif input_mode == 'shuffled_data':
            # input data is shuffled training data
            train = train[:, :-3, :].reshape(-1)
            inds = np.random.shuffle(np.arange(len(train)))
            data = train[inds].reshape(channels, gen_len)
        elif input_mode == 'data':
            data = train[0, :channels, :shift]
            zeros = torch.zeros((channels, gen_len), dtype=torch.int32).cuda()
            data = torch.cat((data, zeros), dim=1)

            # create conditioning data with shift+gen_len length
            '''
            seconds = int(gen_len/self.args.sr_data)*2
            epoch_len = int(self.args.sr_data*0.5)
            cond = []
            for s in range(seconds):
                # choose a class randomly from self.args.num_classes
                cl = np.random.randint(1, self.args.num_classes)
                cond.append(np.array([cl]*epoch_len))

                # uniform distribution between 0.9 and 1
                num_zeros = np.random.randint(int(epoch_len*0.8), epoch_len)
                cond.append(np.zeros((num_zeros)))

            cond = np.concatenate(cond)[:shift+gen_len]
            # replace first epoch with train cond channel
            cond = torch.Tensor(cond).cuda().long()
            cond[:shift] = train[0, -2, :shift]
            '''
            cond = train[:, -2, :shift].reshape(-1)[:shift+gen_len]
            cond = cond.unsqueeze(0).unsqueeze(0)

        elif input_mode == 'frequency':
            # generated data with starting from an input with specific freq
            data = np.random.normal(0, 0.0, (channels, self.args.sr_data*50))
            x = np.arange(shift)/self.args.sr_data
            sine = np.sin(2*np.pi*noise*x)
            sine = (sine - np.mean(sine))/np.std(sine)
            data[0, :shift] = sine
            # data[:, :shift] = self.args.dataset.x_val[:, start:start+shift]
            data = torch.Tensor(data).cuda()
        else:
            raise ValueError('No valid args.generate_input specified.')

        print(data.shape)
        print(cond.shape)

        # recursively generate using the previously defined input
        for t in range(shift, data.shape[1]):
            inputs = data[:, t-shift:t].reshape(1, channels, -1)
            cond_ex = cond[:, :, t-shift:t]
            out = self.forward({'inputs': inputs, 'condition': cond_ex})

            # apply softmax to get probabilities
            out = F.softmax(out, dim=-1)

            # get roulette wheel prediction based on output probabilities (last dimension)
            out = out.reshape(-1, out.shape[-1])

            if sampling == 'roulette':
                out = torch.multinomial(out, 1).reshape(-1)
            elif sampling == 'argmax':
                out = torch.argmax(out, dim=-1).reshape(-1)
            elif sampling == 'top-p':
                # top-p sampling: sample from the smallest set of tokens whose cumulative probability exceeds args.top_p
                # select the smallest set of tokens whose cumulative probability exceeds args.top_p
                sorted_logits, sorted_indices = torch.sort(out, descending=True)
                cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                out[indices_to_remove] = 0
                out = torch.multinomial(out, 1).reshape(-1)

            # switch between IIR, FIR, and purely recursive modes
            if mode == 'IIR':
                data[:, t] = data[:, t] + out
            elif mode == 'FIR':
                output[:, t] = out
            elif mode == 'recursive':
                data[:, t] = out
            else:
                raise ValueError('No valid args.generate_mode specified.')

        if mode == 'FIR':
            data = output

        data = data.cpu().numpy()
        name = 'generated_' + input_mode + mode + sampling + str(noise) + '.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

        return data


class WavenetFullChannelMix(WavenetFullChannel):
    def build_model(self, args):
        super().build_model(args)

        self.mixer = torch.nn.Linear(args.num_channels,
                                     args.num_channels,
                                     bias=False)

    def get_cond(self, data):
        cond = None
        if self.args.cond_channels > 0:
            # cond: B x E x T
            cond_ind = data['condition']
            try:
                inds = torch.squeeze(cond_ind, dim=1)
                cond = self.cond_emb(inds).permute(0, 2, 1)
            except RuntimeError:
                print(cond_ind.shape)
                #print(x.shape)
                #print(inds.shape)
                raise

            # set elements of cond to 0 where cond_ind is 0
            cond = cond * (cond_ind > 0).float()

            # repeat cond across new args.channels dim
            cond = cond.unsqueeze(1).repeat(1, self.args.num_channels, 1, 1)
            cond = cond.reshape(-1, cond.shape[-2], cond.shape[-1])

        return cond

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
        cond = self.get_cond(data)

        # apply embedding to each channel separately
        timesteps = x.shape[-1]
        x = x.permute(0, 2, 1)  # B x T x C
        x = x.reshape(-1, x.shape[-1])  # B*T x C
        x = self.quant_emb[torch.arange(x.shape[-1]), x]  # B*T x C x E

        # reshape back
        x = x.reshape(-1, timesteps, x.shape[-2], x.shape[-1])  # B x T x C x E
        x = x.permute(0, 2, 3, 1)  # B x C x E x T
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # B*C x E x T

        skips = []
        for layer in self.layers:
            x, skip = layer(x, c=cond, causal_pad=causal_pad)
            skips.append(skip)

        if x.shape[-1] != skip.shape[-1]:
            print(x.shape)
            print(skip.shape)

        # sum up skip channels
        skips = sum(skips)

        # reshape to (B, Q, T, C)
        skips = skips.reshape(
            -1, self.args.num_channels, skips.shape[-2], skips.shape[-1])
        skips = skips.permute(0, 2, 3, 1)

        # mix channels
        skips = self.mixer(skips)

        # reshape to (B*C, Q, T)
        skips = skips.permute(0, 3, 1, 2)
        skips = skips.reshape(-1, skips.shape[-2], skips.shape[-1])

        out = self.logits(x, [skips])

        # reshape to get (B*C, Q, T) -> (B, C, T, Q)
        out = out.reshape(
            -1, self.args.num_channels, out.shape[-2], out.shape[-1])
        out = out.permute(0, 1, 3, 2)

        return out


class WavenetContrastiveChannelMix(WavenetFullChannelMix):
    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        logits = self.forward(data)

        # have to make sure this exactly matches the intended targets
        targets = data['targets']
        targets = targets[:, :, -logits.shape[-2]:]

        shape = targets.shape
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])

        all_loss = self.criterion(logits, targets)
        loss = torch.mean(all_loss)

        acc, preds = accuracy(logits, targets)
        acc = torch.mean(acc.float())

        pred_cont = mulaw_inv(preds)
        target_cont = mulaw_inv(targets)

        # compute MSE
        mse = self.mse_loss(pred_cont, target_cont)

        #mse = acc

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss,
                  'valloss/Validation accuracy: ': acc,
                  'trainloss/Train accuracy: ': acc,
                  'trainloss/Training MSE: ': mse,
                  'valloss/Validation MSE: ': mse}

        if self.save_preds:
            pred_cont = pred_cont.detach().cpu().numpy()
            target_cont = target_cont.detach().cpu().numpy()
            all_loss = all_loss.detach().cpu().numpy()

            pred_cont = pred_cont.reshape(shape)
            target_cont = target_cont.reshape(shape)
            all_loss = all_loss.reshape(shape)

            targets = targets.reshape(shape)
            acc = torch.eq(targets[:, :, 1:], targets[:, :, :-1])
            losses['valloss/repeat acc: '] = torch.mean(acc.float())

            train = 'train' if train else 'val'
            # save predictions and targets
            path = os.path.join(self.args.result_dir, train + f'preds{i}.npy')
            np.save(path, pred_cont)
            path = os.path.join(self.args.result_dir, train + f'targets{i}.npy')
            np.save(path, target_cont)
            path = os.path.join(self.args.result_dir, train + f'losses{i}.npy')
            np.save(path, all_loss)

            # save data['sid']
            path = os.path.join(self.args.result_dir, train + f'sid{i}.npy')
            np.save(path, data['sid'].cpu().numpy())

        

        #return losses, pred_cont.reshape(shape), target_cont.reshape(shape)
        return losses, None, None


class WavenetFullChannelMixSemb(WavenetFullChannelMix, WavenetFullTestSemb):
    def build_model(self, args):
        super().build_model(args)
        self.set_covs(args)

    def get_cond(self, data):
        cond = None
        if self.args.cond_channels > 0:
            # cond: B x E x T
            cond_ind = data['condition']
            cond = self.cond_emb(cond_ind.squeeze()).permute(0, 2, 1)

            # set elements of cond to 0 where cond_ind is 0
            cond = cond * (cond_ind > 0).float()

        embs = None
        if self.args.embedding_dim > 0:
            # get subject-specific covariance
            covs = self.covs[data['sid'].reshape(-1)]
            covs = covs.reshape(x.shape[0], -1, covs.shape[1])

            # project to embedding space
            embs = self.subject_emb(covs)
            embs = embs.permute(0, 2, 1)

        if self.args.embedding_dim and self.args.cond_channels:
            # concatenate subject and condition embeddings
            cond = torch.cat([cond, embs], dim=1)

        return cond#, have to integrate fullchannelmix cond logic


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
