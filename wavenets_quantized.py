import torch

from torch.nn import MSELoss, CrossEntropyLoss

from wavenets_full import WavenetFull


class WavenetQuantized(WavenetFull):
    '''
    Implements the full quantized verion of wavenet.
    '''
    def __init__(self, args):
        super(WavenetQuantized, self).__init__(args)

        # CEL is used for quantized loss and MSE is used to get original loss
        self.criterion = CrossEntropyLoss().cuda()
        self.mse = MSELoss().cuda()
        args.num_channels = 1

    def generate_forward(self, inputs, channels):
        '''
        Wrapper around forward function to easily adapt the generate function.
        '''
        inputs = inputs.cpu().numpy()
        inputs = self.args.dataset.quantize(inputs)
        inputs = self.args.dataset.one_hot_encode(inputs)
        inputs = torch.Tensor(inputs).cuda()

        out = self.forward(inputs)[0]
        out = self.dequantize(self.one_hot_decode(out), out.shape[2])

        return out.reshape(channels, -1)

    def loss(self, x, i, train=True):
        '''
        Compute loss for current batch x.
        i: batch index
        train: whether this is train or validation data
        '''
        output, _ = self.forward(x[:, :, :-1])
        timesteps = output.shape[2]

        # compute the cross-entropy loss on the quantized data
        target = self.one_hot_decode(x[:, :, -timesteps:])
        loss = self.criterion(output, target)

        # get back the raw signal from the quantized prediction
        output = self.dequantize(self.one_hot_decode(output), timesteps)

        if train:
            targ = self.args.dataset.get_batch(i, self.args.dataset.x_train_o)
        else:
            targ = self.args.dataset.get_batch(i, self.args.dataset.x_val_o)

        # compute MSE with the non-quantized original signal
        target_real = torch.Tensor(targ).float().cuda()[:, :, -timesteps:]
        real_loss = self.mse(output, target_real)

        # get the dequantized version of the quantized signal
        #target = self.dequantize(target, timesteps)

        return loss, output, target_real, real_loss

    def dequantize(self, x, timesteps):
        '''
        Dequantize a quantized signal x.
        timesteps: length of the examples
        '''
        x = self.args.dataset.dequantize(x.detach().cpu().numpy())
        return torch.Tensor(x).float().cuda().reshape(-1, 1, timesteps)

    def one_hot_decode(self, x):
        '''
        Get the class label from the one-hot encoding.
        '''
        return torch.argmax(x, axis=1)


class WavenetQuantCond(WavenetQuantized):
    