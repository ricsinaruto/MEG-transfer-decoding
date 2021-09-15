from torch.nn import Sequential, Conv1d, Tanh, Sigmoid, ReLU, Dropout

from wavenets_simple import WavenetSimple


class WavenetFull(WavenetSimple):
    '''
    The full wavenet model as described in the original paper,
    but without quantization.
    '''
    def build_model(self, args):
        ch = args.ch_mult * args.num_channels

        self.filter = []
        self.gate = []
        self.conv_skip = []
        self.conv_res = []
        self.tanh = []
        self.sigmoid = []

        self.first_conv = Conv1d(args.num_channels, ch, kernel_size=1)

        # each layer has two convolutions followed by two activation functions
        # followed by two 1x1 convolutions for residual and skip connections
        for rate in args.dilations:
            self.filter.append(Conv1d(ch,
                                      ch,
                                      kernel_size=args.kernel_size,
                                      dilation=rate,
                                      groups=args.groups))
            self.gate.append(Conv1d(ch,
                                    ch,
                                    kernel_size=args.kernel_size,
                                    dilation=rate,
                                    groups=args.groups))
            self.tanh.append(Tanh())
            self.sigmoid.append(Sigmoid())
            self.conv_skip.append(
                Conv1d(ch, ch, kernel_size=1, groups=args.groups))
            self.conv_res.append(
                Conv1d(ch, ch, kernel_size=1, groups=args.groups))

        self.relus = [ReLU(), ReLU()]
        self.last_conv = [Conv1d(ch, ch, kernel_size=1, groups=args.groups),
                          Conv1d(ch, args.num_channels, kernel_size=1)]

        self.filter = Sequential(*self.filter)
        self.gate = Sequential(*self.gate)
        self.conv_skip = Sequential(*self.conv_skip)
        self.conv_res = Sequential(*self.conv_res)
        self.tanh = Sequential(*self.tanh)
        self.sigmoid = Sequential(*self.sigmoid)
        self.relus = Sequential(*self.relus)
        self.last_conv = Sequential(*self.last_conv)

        self.dropout = Dropout(args.p_drop)

    def forward(self, x, sid=None):
        x = self.first_conv(x)
        skips = []

        block = zip(self.filter, self.gate,
                    self.conv_skip, self.conv_res,
                    self.tanh, self.sigmoid)
        for filter_, gate, conv_skip, conv_res, tanh, sigmoid in block:
            xf = filter_(x)
            xg = gate(x)
            xo = tanh(xf)*sigmoid(xg)
            xo = self.dropout(xo)

            xc = conv_res(xo)
            xs = conv_skip(xo)

            x = x[:, :, -xc.shape[2]:] + xc
            skips.append(xs)

        # sum together skip connections
        length = skips[-1].shape[2]
        x = skips[0][:, :, -length:]
        for xs in skips[1:]:
            x = x + xs[:, :, -length:]

        x = self.relus[0](x)
        x = self.last_conv[0](x)
        x = self.relus[1](x)
        x = self.last_conv[1](x)

        return x, None

    def kernel_network_FIR_loop(self, folder, x):
        '''
        Implements loop over the network to get kernel output at each layer.
        '''
        block = zip(
            self.filter, self.gate, self.conv_res, self.tanh, self.sigmoid)

        for i, (filt_, gate, conv_res, tanh, sigmoid) in enumerate(block):
            self.kernel_FIR_plot(folder, x, i, filt_, 'filter')
            self.kernel_FIR_plot(folder, x, i, gate, 'gate')

            xf = filt_(x)
            xg = gate(x)
            xo = tanh(xf)*sigmoid(xg)
            xo = self.dropout(xo)

            xc = conv_res(xo)
            x = x[:, :, -xc.shape[2]:] + xc


class WavenetFullSimple(WavenetSimple):
    '''
    Simple version of wavenet with residual and skip connections.
    '''
    def build_model(self, args):
        super(WavenetFullSimple, self).build_model(args)

        # add two 1x1 convolutions at the end
        del self.last_conv
        self.last_conv = [
            Conv1d(self.ch, self.ch, kernel_size=1, groups=args.groups),
            Conv1d(self.ch, args.num_channels, kernel_size=1)]

        self.last_conv = Sequential(*self.last_conv)
        self.dropout = Dropout(args.p_drop)

    def forward(self, x):
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
        x = self.activation(x)
        x = self.last_conv[1](x)

        return x, None

    def residual(self, data, data_f):
        '''
        This function is needed to make kernel_network_FIR compatible
        with this model. It simply applies the residual connection.
        '''
        return data[:, :, -data_f.shape[2]:] + data_f
