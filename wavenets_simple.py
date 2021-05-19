import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.nn import Sequential, Module, Conv1d, MaxPool1d, Identity
from torch.nn import MSELoss, LogSoftmax
from torch.optim import Adam
import torch.nn.functional as F

from scipy.signal import welch
from scipy import signal
from scipy.io import savemat


class WavenetSimple(Module):
    '''
    Implements a simplified version of wavenet no padding.
    '''
    def __init__(self, args):
        super(WavenetSimple, self).__init__()
        self.args = args
        self.timesteps = args.timesteps
        self.build_model(args)

        self.criterion = MSELoss().cuda()
        self.activation = self.args.activation
        if self.args.linear:
            self.activation = Identity()

    def build_model(self, args):
        '''
        Specify the layers of the model.
        '''
        self.ch = args.ch_mult * args.num_channels
        modules = []

        # 1x1 convolution to project to hidden channels
        self.first_conv = Conv1d(args.num_channels, self.ch, kernel_size=1)

        # each layer consists of a dilated convolution
        # followed by a nonlinear activation
        for rate in args.dilations:
            # modules.append(ConstantPad1d((rate, 0), 0.0))
            modules.append(Conv1d(self.ch,
                                  self.ch,
                                  kernel_size=args.kernel_size,
                                  dilation=rate,
                                  groups=args.groups))

        # 1x1 convolution to go back to original channel dimension
        self.last_conv = Conv1d(self.ch, args.num_channels, kernel_size=1)

        self.cnn_layers = Sequential(*modules)

    def forward(self, x):
        x = self.first_conv(x)

        for layer in self.cnn_layers:
            x = self.activation(layer(x))

        return self.last_conv(x), x

    def loss(self, x, i=0, train=True):
        '''
        If timesteps is bigger than 1 this loss can be used to predict any
        timestep in the future directly, e.g. t+2 or t+5, etc.
        '''
        output, _ = self.forward(x[:, :, :-self.timesteps])
        target = x[:, :, -output.shape[2]:]
        loss = self.criterion(output, target)

        return loss, output, target, None

    def repeat_loss(self, batch):
        '''
        Baseline loss for repeating the same timestep for future.
        '''
        start = int(batch.shape[2] / 2)
        return self.criterion(batch[:, :, start:-1], batch[:, :, start + 1:])

    def ar_loss(self, output, target):
        '''
        Applies the MSE loss between output and target.
        '''
        return self.criterion(output, target)

    def channel_output(self, x, num_l, num_c):
        '''
        Compute the output for a specific layer num_l and channel num_c.
        '''
        x = self.layer_output(x, num_l)
        return -torch.mean(x[:, num_c, :])

    def layer_output(self, x, num_l):
        '''
        Compute the output for a specific layer num_l.
        '''
        x = self.first_conv(x)
        for i in range(num_l + 1):
            x = self.cnn_layers[i](x)
            if i < num_l:
                x = self.activation(x)
        return x

    def run_kernel(self, x, num_l, num_kernel):
        '''
        Compute the output of a specific kernel num_kernel
        in a specific layer num_l to input x.
        '''
        inp_filt = int(num_kernel/self.args.ch_mult)
        out_filt = num_kernel % self.args.ch_mult

        # deconstruct convolution to get specific kernel output
        x = F.conv1d(x[:, inp_filt:inp_filt + 1, :],
                     self.cnn_layers[num_l].weight[
                        out_filt:out_filt + 1, inp_filt:inp_filt + 1, :],
                     self.cnn_layers[num_l].bias[out_filt:out_filt + 1],
                     self.cnn_layers[num_l].stride,
                     self.cnn_layers[num_l].padding,
                     self.cnn_layers[num_l].dilation)

        return x

    def kernel_output(self, x, num_l, num_f):
        '''
        Compute the output for a specific layer num_l and kernel num_f.
        '''
        x = self.layer_output(x, num_l-1)
        x = self.activation(x)
        x = self.run_kernel(x, num_l, num_f)

        return -torch.mean(x)

    def plot_welch(self, x, ax, args, i):
        '''
        Compute and plot (on ax) welch spectra of x.
        '''
        f, Pxx_den = welch(x, args.sr_data, nperseg=4*args.sr_data)
        ax.plot(f, Pxx_den)
        for freq in args.freqs:
            ax.axvline(x=freq, color='red')

    def analyse_kernels(self):
        '''
        Learn input for each kernel to see what patterns they are sensitive to.
        '''
        folder = os.path.join(self.args.result_dir, 'kernel_analysis')
        if not os.path.isdir(folder):
            os.mkdir(folder)

        indiv = self.args.individual
        func = self.kernel_output if indiv else self.channel_output
        num_filters = self.args.ch_mult**2 if indiv else self.args.ch_mult
        figsize = (15, 10*num_filters)

        # loop over all layers and kernels in the model
        for num_layer in range(len(self.args.dilations)):
            fig, axs = plt.subplots(num_filters+1, figsize=figsize)

            for num_filter in range(num_filters):
                # optimize input signal for a given kernel
                batch = torch.randn(
                    (1, 1, self.args.rf*20-self.args.timesteps),
                    requires_grad=True,
                    device='cuda')
                optimizer = Adam([batch], lr=self.args.anal_lr)

                for epoch in range(self.args.anal_epochs):
                    # if we don't clamp the input would explode
                    with torch.no_grad():
                        batch.clamp_(-5, 5)

                    # add L2 regularization
                    loss = torch.norm(batch) * self.args.norm_coeff

                    loss += func(batch, num_layer, num_filter)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if not epoch % 20:
                        print('Filter loss: ', loss.item())

                # save learned input to disk
                inputs = batch.view(-1).detach().cpu().numpy()
                name = str(num_layer) + '_' + str(num_filter) + '.mat'
                savemat(os.path.join(folder, name), {'X': inputs})

                # compute fft of learned input
                self.plot_welch(inputs, axs[num_filter], self.args, num_layer)

            name = '_indiv' if indiv else ''
            filename = os.path.join(
                folder, 'layer' + str(num_layer) + name + '_freq.svg')
            fig.savefig(filename, format='svg', dpi=2400)
            plt.close('all')

    def generate_forward(self, inputs, channels):
        '''
        Wrapper around forward function to easily adapt the generate function.
        '''
        return self.forward(inputs)[0].detach().reshape(channels, -1)

    def generate(self):
        '''
        Recursively generate with a trained model in various ways.
        '''
        self.eval()
        input_mode = self.args.generate_input
        mode = self.args.generate_mode
        noise = self.args.generate_noise
        channels = self.args.num_channels
        shift = self.args.rf
        gen_len = self.args.generate_length

        output = torch.normal(0.0, noise, size=(channels, gen_len)).cuda()

        if input_mode == 'gaussian_noise':
            # input is gaussian noise
            data = torch.normal(0.0, noise, size=(channels, gen_len)).cuda()
        elif input_mode == 'shuffled_data':
            # input data is shuffled training data
            train = self.args.dataset.x_train[0, :]
            data = np.random.choice(train, (channels, gen_len))
            data = torch.Tensor(data).cuda() * noise
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

        # recursively generate using the previously defined input
        for t in range(shift, data.shape[1]):
            inputs = data[:, t-shift:t].reshape(1, channels, -1)
            out = self.generate_forward(inputs, channels)

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
        name = 'generated_' + input_mode + mode + str(noise) + '.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

        return data

    def randomize_kernel_input(self, data):
        '''
        Randomize the input for a specific kernel for kernel_network_FIR.
        '''
        input_data = data.detach().cpu().numpy()
        choosing_data = data.detach().cpu().numpy()
        for c in range(input_data.shape[1]):
            choose_channel = choosing_data[:, c, :].reshape(-1)
            length = choose_channel.shape[0]
            input_data[:, c, :] = np.random.choice(choose_channel, (1, length))
        return torch.Tensor(input_data).cuda()

    def residual(self, data, data_f):
        return data_f

    def kernel_network_FIR(self,
                           folder='kernels_network_FIR',
                           generated_data=None):
        '''
        Get FIR properties for each kernel by running the whole network.
        '''
        folder = os.path.join(self.args.result_dir, folder)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        gen_len = self.args.generate_length

        # data is either drawn from gaussian or passed as argument to this func
        data = np.random.normal(0, self.args.generate_noise, (gen_len))
        if generated_data is not None:
            data = generated_data
        data = torch.Tensor(data).cuda().reshape(1, 1, -1)

        data = self.first_conv(data)

        # loop over whole network
        self.kernel_network_FIR_loop(folder, data)

    def kernel_network_FIR_loop(self, folder, data):
        '''
        Implements loop over the network to get kernel output at each layer.
        '''
        for i, layer in enumerate(self.cnn_layers):
            self.kernel_FIR_plot(folder, data, i)

            # compute output of current layer
            data_f = self.activation(layer(data))
            data = self.residual(data, data_f)

    def kernel_FIR_plot(self, folder, data, i):
        '''
        Plot FIR response of kernels in current layer (i) to input data.
        '''
        input_data = torch.Tensor(data.cpu()).cuda()

        num_plots = min(self.args.kernel_limit, self.args.ch_mult**2)
        fig, axs = plt.subplots(num_plots+1, figsize=(20, num_plots*3))

        filter_outputs = []
        for k in range(num_plots):
            x = self.run_kernel(input_data, i, k)
            x = x.detach().cpu().numpy().reshape(-1)
            filter_outputs.append(x)

            # compute fft of kernel output
            self.plot_welch(x, axs[k], self.args, i)

        filter_outputs = np.array(filter_outputs)
        path = os.path.join(folder, 'layer' + str(i) + '.mat')
        savemat(path, {'X': filter_outputs})

        filename = os.path.join(folder, 'layer' + str(i) + '.svg')
        fig.savefig(filename, format='svg', dpi=2400)
        plt.close('all')

    def kernel_network_IIR(self):
        '''
        Get IIR properties for each kernel by running the whole network.
        '''
        data = self.generate()
        self.kernel_network_FIR('kernels_network_IIR', data)

    def plot_kernels(self):
        '''
        Plot kernels and frequency response of kernels.
        '''
        folder = os.path.join(self.args.result_dir, 'kernels')
        if not os.path.isdir(folder):
            os.mkdir(folder)

        ks = self.args.kernel_size

        all_kernels = []
        for i, layer in enumerate(self.cnn_layers):
            kernels = layer.weight.detach().cpu().numpy()
            all_kernels.append(kernels)
            kernels = kernels.reshape(-1, ks)

            num_plots = kernels.shape[0]+1
            fig, axs = plt.subplots(num_plots, figsize=(5, num_plots*3))
            fig_freq, axs_freq = plt.subplots(num_plots,
                                              figsize=(20, num_plots*3))

            # plot individual kernels
            for k in range(num_plots-1):
                axs[k].plot(kernels[k], linewidth=0.5)

                # create filter from dilated kernel
                filter_coeff = []
                spacing = layer.dilation[0]-1
                for j in range(ks):
                    filter_coeff.append(kernels[k, -j])
                    filter_coeff.extend([0] * spacing)
                filter_coeff.append(kernels[k, 0])
                filter_coeff = np.array(filter_coeff)

                # filter_coeff = np.append(1, filter_coeff)
                w, h = self.scipy_freqz(filter_coeff, i)
                axs_freq[k].plot(w, np.abs(h))

                for freq in self.args.freqs:
                    axs_freq[k].axvline(x=freq, color='red')

            filename = os.path.join(folder, 'kernel_layer' + str(i) + '.svg')
            fig.savefig(filename, format='svg', dpi=2400)
            plt.close('all')

            filename = os.path.join(folder, 'kernel_freq' + str(i) + '.svg')
            fig_freq.savefig(filename, format='svg', dpi=2400)
            plt.close('all')

        # self.multiplied_kernels(all_kernels)

    def scipy_freqz(self, filter_coeff, i):
        '''
        Helper function for the signal.freqz function.
        '''
        sr = self.args.sr_data
        return signal.freqz(b=filter_coeff, fs=sr, worN=5*sr)

    def multiplied_kernels(self, all_kernels):
        pass
        '''
        Multiply together kernels along each path from input receptive field
        all_kernels = all_kernels[::-1]
        hid_ch = all_kernels[0].shape[0]
        total_channels = hid_ch**(len(all_kernels)+1)
        num_plots = total_channels+1 if total_channels < 201 else 201
        fig, axs = plt.subplots(num_plots, figsize=(20, num_plots))

        plots = []
        for out_ch in range(hid_ch):
            prev_kernel = all_kernels[0][out_ch, :, :]
            for kernel in all_kernels[1:]:

                all_ch_values = []
                for i, prev_ch in enumerate(prev_kernel):
                    for kernel_ch in kernel[i % hid_ch, :, :]:
                        values = []
                        for prev_value in prev_ch:
                            val = prev_value*kernel_ch
                            values.extend(val)
                        all_ch_values.append(values)
                prev_kernel = all_ch_values

            plots.extend(prev_kernel)

        axs[0].plot(np.sum(np.array(plots), axis=0))
        for i, ch in enumerate(plots):
            if i < 200:
                axs[i+1].plot(ch)

        filename = os.path.join(self.args.result_dir,
                                'kernels',
                                'multiplied_kernels.svg')
        plt.savefig(filename, format='svg', dpi=2400)
        plt.close('all')
        '''

    def network_output(self):
        pass
        '''
        # compute histogram
        x = self.args.dataset.x_val[0, :10000]

        outputs = []
        x = torch.Tensor(x).cuda()
        x = x.reshape(1, 1, -1)

        outputs.append(x.detach().cpu().numpy()[0, 0, :])
        x = self.first_conv(x)
        outputs.append(x.detach().cpu().numpy()[0, 0, :])

        for layer in self.cnn_layers:
            x = layer(x)
            if not self.args.linear:
                x = self.selu(x)

            outputs.append(x.detach().cpu().numpy()[0, 0, :])

        x = self.last_conv(x)
        outputs.append(x.detach().cpu().numpy()[0, 0, :])



        fig, axs = plt.subplots(len(data), figsize=(10, len(data)*3))

        for i, out in enumerate(data):
            axs[i].hist(out, bins=1000)

        filename = os.path.join(self.args.result_dir,
                                'kernels',
                                'layer_histograms.svg')
        fig.savefig(filename, format='svg', dpi=2400)
        plt.close('all')
        '''

    def kernelxyz(self):
        pass
        '''
        # compute xyz for each kernel
        triplets = self.kernelxyz(data)
        fig, axs = plt.subplots(len(triplets), subplot_kw={'projection': '3d'})

        for i, xyz in enumerate(triplets):
            x, y = np.meshgrid(xyz[0], xyz[1])
            axs[i].scatter3D(xyz[0], xyz[1], xyz[2], c=xyz[2],cmap=cm.coolwarm)
            axs[i].plot3D(xyz[0], xyz[1], xyz[2])

        path = os.path.join(self.args.result_dir, 'kernels', 'kernelxyz')
        pickle.dump(triplets, open(path, 'wb'))
        filename = os.path.join(self.args.result_dir,
                                'kernels',
                                'kernelxyz.svg')
        fig.savefig(filename, format='svg', dpi=2400)
        plt.close('all')
                triplets = []
        x = torch.Tensor(x).cuda()
        x = x.reshape(1, 1, -1)
        ch = self.args.ch_mult

        x = self.first_conv(x)

        for layer in self.cnn_layers:
            inp = x
            x = layer(x)

            for inp_ch in range(ch):
                for out_ch in range(ch):
                    out = F.conv1d(inp[:, inp_ch:inp_ch + 1, :],
                        layer.weight[out_ch:out_ch + 1, inp_ch:inp_ch + 1, :],
                                   None,
                                   layer.stride,
                                   layer.padding,
                                   layer.dilation)

                    out = self.selu(out)
                    out = out.reshape(-1).detach().cpu().numpy()
                    inp1 = inp[0, inp_ch, :out.shape[0]].detach().cpu().numpy()
                    inp2 = inp[0, inp_ch,-out.shape[0]:].detach().cpu().numpy()

                    triplets.append((inp1, inp2, out))

            if not self.args.linear:
                x = self.selu(x)

        return triplets
        '''


class WavenetMultistep(WavenetSimple):
    '''
    A variant of the simple Wavenet where multiple timesteps are predicted
    for each timestep in the input serially.
    '''
    def __init__(self, args):
        super(WavenetMultistep, self).__init__(args)
        self.input_len = args.sample_rate - args.timesteps
        self.padding = torch.zeros(self.input_len).float().cuda()

    def forward(self, x):
        # put paddings after each timestep for future timestep predictions
        # if we want to predict 3 timesteps in the future this is the input:
        # t1 0 0 t2 0 0 t3 0 0 t4 0 0 t5...
        paddings = self.padding.repeat(x.shape[0], x.shape[1], 1)
        paddings = [paddings for i in range(self.timesteps-1)]
        x = self.interleave([x] + paddings)

        x = self.first_conv(x)

        # loop over convolution layers
        for i, layer in enumerate(self.cnn_layers):
            # t+1 prediction
            x_t1 = layer(x[:, :, ::self.timesteps])

            # activations is a list of outputs for future timesteps: t1, t2...
            activations = [x_t1]

            # get output for each future timestep with a separate convolution
            # this loop should be parallelized
            for t in range(1, self.timesteps):
                # dilation rate adapted to padded input
                d = self.cnn_layers[i].dilation[0]
                start = t - d
                if d > t:
                    d = t + (d - t) * (t + 1)
                    start = 0

                # input to convolution should be only every t input
                indices = []
                for index in range(x.shape[2]):
                    if index % self.timesteps <= t and index >= start:
                        indices.append(index)

                out = F.conv1d(x[:, :, indices],
                               weight=self.cnn_layers[i].weight,
                               bias=self.cnn_layers[i].bias,
                               stride=t+1,
                               padding=self.cnn_layers[i].padding,
                               dilation=d,
                               groups=self.cnn_layers[i].groups)

                # truncate to correct shape by removing early elements
                out = out[:, :, -x_t1.shape[2]:]
                activations.append(out)

            # interleave the outputs for all future timesteps
            x = self.interleave(activations)

            # apply activation function
            x = self.activation(x)

        return self.last_conv(x), x

    def interleave(self, x):
        '''
        Interleave the elements of x.
        '''
        if len(x) == 1:
            return x[0]
        x = torch.stack(tuple(x), dim=3)
        x = x.view(x.shape[0], x.shape[1], -1)
        return x

    def loss(self, x, i=0, train=True):
        output, _ = self.forward(x[:, :, :-self.timesteps])

        # construct target by interleaving future timesteps
        targets = []
        for i in range(1, self.timesteps + 1):
            targets.append(x[:, :, i:x.shape[2]-self.timesteps+i])
        targets = self.interleave(targets)
        targets = targets[:, :, -output.shape[2]:]

        loss = self.criterion(output, targets)

        # calculate loss separately for t+1 and t+2 predictions
        loss2 = None
        if self.timesteps == 2:
            # t+1
            targ = targets[:, :, ::2]
            out = output[:, :, ::2]
            loss1 = self.criterion(targ, out)

            # t+2
            targ = targets[:, :, 1::2]
            out = output[:, :, 1::2]
            loss2 = self.criterion(targ, out)

            # since t+1 is easier to predict, it has lower loss so we reweight
            loss = 10 * loss1 + loss2

        return loss, output, targets, loss2


class ConvPoolNet(WavenetSimple):
    '''
    Simple convolutional model using max pooling instead of dilation.
    '''
    def build_model(self, args):
        super(ConvPoolNet, self).build_model(args)

        # add the maxpool layers to the model
        modules = [MaxPool1d(kernel_size=2, stride=2) for r in args.dilations]
        self.maxpool_layers = Sequential(*modules)

    def forward(self, x):
        x = self.first_conv(x)

        for conv, pool in zip(self.cnn_layers, self.maxpool_layers):
            x = self.activation(pool(conv(x)))

        return self.last_conv(x), x

    def kernel_network_FIR_loop(self, folder, data):
        '''
        Implements loop over the network to get kernel output at each layer.
        '''
        layers = zip(self.cnn_layers, self.maxpool_layers)
        for i, (conv, pool) in enumerate(layers):
            self.kernel_FIR_plot(folder, data, i)

            # compute output of current layer
            data_f = self.activation(pool(conv(data)))
            data = self.residual(data, data_f)

    def layer_output(self, x, num_l):
        '''
        Compute the output for a specific layer num_l.
        '''
        x = self.first_conv(x)
        for i in range(num_l + 1):
            x = self.cnn_layers[i](x)
            if i < num_l:
                x = self.maxpool_layers[i](x)
                x = self.activation(x)
        return x

    def kernel_output(self, x, num_l, num_f):
        '''
        Compute the output for a specific layer num_l and kernel num_f.
        '''
        x = self.layer_output(x, num_l-1)
        x = self.maxpool_layers[num_l-1](x)
        x = self.activation(x)
        x = self.run_kernel(x, num_l, num_f)

        return -torch.mean(x)

    def plot_welch(self, x, ax, args, i):
        '''
        Compute and plot (on ax) welch spectra of x.
        The sampling rate in each layer (i) is halved.
        '''
        sr = args.sr_data / 2**i
        f, Pxx_den = welch(x, sr, nperseg=4*sr)
        ax.plot(f, Pxx_den)
        for freq in args.freqs:
            ax.axvline(x=freq, color='red')

    def scipy_freqz(self, filter_coeff, i):
        '''
        Helper function for the signal.freqz function. (i: layer index)
        '''
        sr = int(self.args.sr_data / 2**i)
        return signal.freqz(b=filter_coeff, fs=sr, worN=5*sr)


class Conv1PoolNet(ConvPoolNet):
    '''
    Simple convolutional model using max pooling only in first layer.
    '''

    def forward(self, x):
        x = self.first_conv(x)
        x = self.activation(self.maxpool_layers[0](self.cnn_layers[0](x)))

        for conv in self.cnn_layers:
            x = self.activation(conv(x))

        print(x.shape)
        return self.last_conv(x), x


class WavenetCPC(WavenetSimple):
    '''
    Implements the contrastive predictive coding (CPC) loss for the
    simple wavenet. This loss compares the output to ground truth and
    N distractors through softmax.
    '''
    def __init__(self, args):
        super(WavenetCPC, self).__init__(args)
        self.softmax = LogSoftmax(dim=3)
        self.mse = MSELoss(reduction='none')

    def loss(self, x, i=0, train=True):
        output, _ = self.forward(x[:, :, :-self.timesteps])
        num_out = output.shape[2]
        num_samples = self.args.num_samples_CPC

        targets = []
        numbers = np.arange(x.shape[2])
        # create special target with distractors for each output timestep
        for i in range(num_out):
            current = i + x.shape[2] - num_out + self.timesteps - 1
            without_current = np.delete(numbers, current)
            inds = np.random.permutation(without_current)[:num_samples]
            targets.append(x[:, :, list(inds) + [current]])

        targets = torch.stack(targets, dim=2)
        output = torch.unsqueeze(output, 3)
        output = torch.repeat_interleave(output, num_samples+1, 3)

        # compute CPC loss with MSE as a measure of similarity
        mse = self.mse(output, targets)
        scaled_mse = (1-mse) / self.args.k_CPC
        loss = -torch.mean(self.softmax(scaled_mse)[:, :, :, -1])
        loss2 = torch.mean(mse[:, :, :, -1])

        target = x[:, :, -output.shape[2]:]
        return loss, output[:, :, :, 0], target, loss2
