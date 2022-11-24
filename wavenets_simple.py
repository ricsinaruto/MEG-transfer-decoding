import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import pickle

from torch.nn import Sequential, Module, Conv1d, MaxPool1d, Dropout
from torch.nn import MSELoss, LogSoftmax, Dropout2d, Embedding, Linear
from torch.optim import Adam
import torch.nn.functional as F

from scipy.signal import welch
from scipy import signal
from scipy.io import savemat


class WavenetSimple(Module):
    '''
    Implements a simplified version of wavenet without padding.
    '''
    def __init__(self, args):
        super(WavenetSimple, self).__init__()
        self.args = args
        self.inp_ch = args.num_channels
        self.out_ch = args.num_channels
        self.kernel_inds = []

        self.timesteps = args.timesteps
        self.build_model(args)

        self.criterion = MSELoss().cuda()
        self.activation = self.args.activation

        # add dropout to each layer
        self.dropout2d = Dropout2d(args.p_drop)

    def loaded(self, args):
        '''
        When model is loaded from file, assign the new args object.
        '''
        self.kernel_inds = []
        self.args = args
        self.shuffle_embeddings = False
        self.dropout2d = Dropout2d(args.p_drop)

    def build_model(self, args):
        '''
        Specify the layers of the model.
        '''
        self.ch = int(args.ch_mult * self.inp_ch)

        conv1x1_groups = args.conv1x1_groups
        modules = []

        # 1x1 convolution to project to hidden channels
        self.first_conv = Conv1d(
            self.inp_ch, self.ch, kernel_size=1, groups=conv1x1_groups)

        # each layer consists of a dilated convolution
        # followed by a nonlinear activation
        for rate in args.dilations:
            modules.append(Conv1d(self.ch,
                                  self.ch,
                                  kernel_size=args.kernel_size,
                                  dilation=rate,
                                  groups=args.groups))

        # 1x1 convolution to go back to original channel dimension
        self.last_conv = Conv1d(
            self.ch, self.out_ch, kernel_size=1, groups=conv1x1_groups)

        self.cnn_layers = Sequential(*modules)

        self.subject_emb = Embedding(args.subjects, args.embedding_dim)

    def get_weight_nograd(self, layer):
        return layer.weight.detach().clone().requires_grad_(False)

    def get_weight(self, layer):
        return layer.weight

    def get_weights(self, grad=False):
        '''
        Return a list of all weights in the model.
        '''
        get_weight = self.get_weight if grad else self.get_weight_nograd

        weights = [get_weight(layer) for layer in self.cnn_layers]
        weights.append(get_weight(self.first_conv))
        weights.append(get_weight(self.last_conv))

        return weights

    def save_embeddings(self):
        '''
        Save subject embeddings.
        '''
        weights = {'X': self.subject_emb.weight.detach().cpu().numpy()}
        savemat(os.path.join(self.args.result_dir, 'sub_emb.mat'), weights)

    def dropout(self, x):
        '''
        Applies 2D dropout to 1D data by unsqueezeing.
        '''
        if self.args.dropout2d_bad:
            x = self.dropout2d(x)
        else:
            x = torch.unsqueeze(x, 3)
            x = self.dropout2d(x)
            x = x[:, :, :, 0]

        return x

    def forward4(self, x, sid=None):
        '''
        Only use the first few layers of Wavenet.
        '''
        x = self.first_conv(x)

        # the layer from which we should get the output is
        # automatically calculated based on the receptive field
        lnum = int(np.log(self.args.rf) / np.log(self.args.kernel_size)) - 1
        for i, layer in enumerate(self.cnn_layers):
            x = self.activation(self.dropout(layer(x)))
            if i == lnum:
                break

        return self.last_conv(x), x

    def forward(self, x, sid=None):
        '''
        Run a forward pass through the network.
        '''
        x = self.first_conv(x)

        for layer in self.cnn_layers:
            x = self.activation(self.dropout(layer(x)))

        return self.last_conv(x), x

    def end(self):
        pass

    def loss(self, x, i=0, sid=None, train=True, criterion=None):
        '''
        If timesteps is bigger than 1 this loss can be used to predict any
        timestep in the future directly, e.g. t+2 or t+5, etc.
        sid: subject index
        '''
        output, _ = self.forward(x[:, :, :-self.timesteps], sid)
        target = x[:, :, -output.shape[2]:]
        if criterion is None:
            loss = self.criterion(output, target)
        else:
            loss = criterion(output, target)

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/Validation loss: ': loss,
                  'valloss/saveloss/none': loss}

        return losses, output, target

    def repeat_loss(self, batch):
        '''
        Baseline loss for repeating the same timestep for future.
        '''
        start = int(batch.shape[2] / 2)
        loss = self.criterion(batch[:, :, start:-1], batch[:, :, start + 1:])

        return {'valloss/Repeat loss: ': loss}

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

    def layer_output(self, x, num_l, sid=None):
        '''
        Compute the output for a specific layer num_l.
        '''
        x = self.first_conv(x)
        for i in range(num_l + 1):
            x = self.cnn_layers[i](x)
            if i < num_l:
                x = self.activation(self.dropout(x))
        return x

    def run_kernel(self, x, layer, num_kernel):
        '''
        Compute the output of a specific kernel num_kernel
        in a specific layer (layer) to input x.
        '''
        # TODO: current assumption is that the network is fully depthwise
        chid = self.args.channel_idx
        ch = self.args.ch_mult

        # input and output filter indices
        out_filt = int(num_kernel/ch) + chid * ch
        inp_filt = num_kernel % ch

        # select specific channel
        x = x[:, chid*ch:(chid+1)*ch, :]

        # deconstruct convolution to get specific kernel output
        x = F.conv1d(x[:, inp_filt:inp_filt + 1, :],
                     layer.weight[
                        out_filt:out_filt + 1, inp_filt:inp_filt + 1, :],
                     layer.bias[out_filt:out_filt + 1],
                     layer.stride,
                     layer.padding,
                     layer.dilation)

        return x

    def kernel_output_all(self, x, num_l, num_f, sid=None):
        '''
        Compute the output for a specific layer num_l and kernel num_f.
        '''
        x = self.layer_output(x, num_l-1, sid)
        x = self.activation(self.dropout(x))
        x = self.run_kernel_multi(x, self.cnn_layers[num_l], num_f)

        return x.detach().cpu()

    def kernel_output(self, x, num_l, num_f):
        '''
        Compute the output for a specific layer num_l and kernel num_f.
        '''
        x = self.kernel_output_all(x, num_l, num_f)
        return -torch.mean(x)

    def plot_welch(self, x, ax, sr=1):
        '''
        Compute and plot (on ax) welch spectra of x.
        '''
        f, Pxx_den = welch(x, self.args.sr_data, nperseg=4*self.args.sr_data)
        ax.plot(f, Pxx_den)
        for freq in self.args.freqs:
            ax.axvline(x=freq, color='red')

    def kernelPFI(self, data, sid=None):
        if not self.kernel_inds:
            for _ in range(len(self.args.dilations)):
                for f in range(self.args.kernel_limit):
                    inds1 = random.randint(0, self.ch-1)
                    inds2 = random.randint(0, self.ch-1)
                    self.kernel_inds.append((inds1, inds2))

        outputs = []
        for num_layer in range(len(self.args.dilations)):
            for num_filter in range(self.args.kernel_limit):
                ind = num_layer*self.args.kernel_limit + num_filter
                x = self.kernel_output_all(data, num_layer, ind, sid)
                outputs.append(x)

        return outputs

    def run_kernel_multi(self, x, layer, num_kernel):
        '''
        Compute the output of a specific kernel num_kernel
        in a specific layer (layer) to input x.
        '''
        # input and output filter indices
        if self.args.kernel_inds:
            out_filt = self.args.kernel_inds[num_kernel][0]
            inp_filt = self.args.kernel_inds[num_kernel][1]
        elif self.kernel_inds:
            out_filt = self.kernel_inds[num_kernel][0]
            inp_filt = self.kernel_inds[num_kernel][1]
        else:
            out_filt = random.randint(0, self.ch-1)
            inp_filt = random.randint(0, self.ch-1)

        # deconstruct convolution to get specific kernel output
        x = F.conv1d(x[:, inp_filt:inp_filt + 1, :],
                     layer.weight[
                        out_filt:out_filt + 1, inp_filt:inp_filt + 1, :],
                     layer.bias[out_filt:out_filt + 1],
                     layer.stride,
                     layer.padding,
                     layer.dilation)

        return x

    def analyse_kernels(self):
        '''
        Learn input for each kernel to see what patterns they are sensitive to.
        '''
        self.eval()
        indiv = self.args.individual
        hid_ch = self.args.ch_mult * self.args.num_channels
        input_len = self.args.generate_length

        folder = os.path.join(self.args.result_dir, 'kernel_analysis')
        if not os.path.isdir(folder):
            os.mkdir(folder)

        func = self.kernel_output if indiv else self.channel_output
        num_filters = self.args.kernel_limit if indiv else hid_ch
        figsize = (15, 10*num_filters)

        # compute inverse of standardization
        path = os.path.join('/'.join(self.args.dump_data.split('/')[:-1]),
                            'standardscaler')
        with open(path, 'rb') as file:
            norm = pickle.load(file)

        losses = []
        # loop over all layers and kernels in the model
        for num_layer in range(len(self.args.dilations)):
            fig, axs = plt.subplots(num_filters+1, figsize=figsize)

            for num_filter in range(num_filters):
                # optimize input signal for a given kernel
                batch = torch.randn(
                    (1, self.args.num_channels, input_len),
                    requires_grad=True,
                    device='cuda')
                optimizer = Adam([batch], lr=self.args.anal_lr)

                losses_in = []
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
                    losses_in.append(loss.item())

                # save learned input to disk
                inputs = batch.squeeze().detach().cpu().numpy()
                inputs = norm.inverse_transform(inputs.T).T
                name = str(num_layer) + '_' + str(num_filter) + '.mat'
                savemat(os.path.join(folder, name), {'X': inputs})

                # compute fft of learned input
                #self.plot_welch(inputs, axs[num_filter], num_layer)
                losses.append(np.array(losses_in))

            '''
            name = '_indiv' if indiv else ''
            filename = os.path.join(
                folder, 'layer' + str(num_layer) + name + '_freq.svg')
            fig.savefig(filename, format='svg', dpi=2400)
            plt.close('all')
            '''

        # save losses
        savemat(os.path.join(folder, 'losses.mat'), {'L': losses})

    def generate_forward(self, inputs, channels):
        '''
        Wrapper around forward function to easily adapt the generate function.
        '''
        return self.forward(inputs)[0].detach().reshape(channels)

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
        elif input_mode == 'none':
            data = torch.normal(0.0, noise, size=(channels, shift))
            data = torch.cat((data, torch.zeros((channels, gen_len))), dim=1)
            data = data.cuda()
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

        # compute welch spectra for each channel
        fig, axs = plt.subplots(channels+1, figsize=(15, channels*6))
        for ch in range(channels):
            self.plot_welch(data[ch, :], axs[ch], ch)

        name = 'generated_freqs_' + input_mode + mode + str(noise) + '.svg'
        path = os.path.join(self.args.result_dir, name)
        fig.savefig(path, format='svg', dpi=1200)
        plt.close('all')

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
        self.eval()
        name = folder + 'ch' + str(self.args.channel_idx)
        folder = os.path.join(self.args.result_dir, name)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # data is either drawn from gaussian or passed as argument to this func
        shape = (self.args.num_channels, self.args.generate_length)
        data = np.random.normal(0, self.args.generate_noise, shape)
        if generated_data is not None:
            data = generated_data
        data = torch.Tensor(data).cuda().reshape(1, self.args.num_channels, -1)

        data = self.first_conv(data)

        # loop over whole network
        self.kernel_network_FIR_loop(folder, data)

    def kernel_network_FIR_loop(self, folder, data):
        '''
        Implements loop over the network to get kernel output at each layer.
        '''
        for i, layer in enumerate(self.cnn_layers):
            self.kernel_FIR_plot(folder, data, i, layer)

            # compute output of current layer
            data_f = self.activation(self.dropout(layer(data)))
            data = self.residual(data, data_f)

    def kernel_FIR_plot(self, folder, data, i, layer, name='conv'):
        '''
        Plot FIR response of kernels in current layer (i) to input data.
        '''
        num_plots = self.args.kernel_limit
        fig, axs = plt.subplots(num_plots+1, figsize=(20, num_plots*3))

        multi = self.args.groups == 1
        kernel_func = self.run_kernel_multi if multi else self.run_kernel

        filter_outputs = []
        for k in range(num_plots):
            x = kernel_func(data, layer, k)
            x = x.detach().cpu().numpy().reshape(-1)
            filter_outputs.append(x)

            # compute fft of kernel output
            self.plot_welch(x, axs[k], i)

        filter_outputs = np.array(filter_outputs)
        path = os.path.join(folder, name + str(i) + '.mat')
        savemat(path, {'X': filter_outputs})

        filename = os.path.join(folder, name + str(i) + '.svg')
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
        self.eval()
        ks = self.args.kernel_size
        mode = self.args.generate_mode
        chid = self.args.channel_idx
        ch = self.args.ch_mult

        name = 'kernels' + mode + 'ch' + str(chid)
        folder = os.path.join(self.args.result_dir, name)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        weights = {'X': self.first_conv.weight.detach().cpu().numpy()}
        savemat(os.path.join(self.args.result_dir, 'first_conv.mat'), weights)
        weights = {'X': self.last_conv.weight.detach().cpu().numpy()}
        savemat(os.path.join(self.args.result_dir, 'last_conv.mat'), weights)

        func = self.scipy_freqz_IIR if mode == 'IIR' else self.scipy_freqz_FIR

        all_kernels = []
        for i, layer in enumerate(self.cnn_layers):
            kernels = layer.weight.detach().cpu().numpy()
            kernels = kernels[chid*ch:(chid+1)*ch, :, :]
            all_kernels.append(kernels)
            kernels = kernels.reshape(-1, ks)

            num_plots = min(kernels.shape[0]+1, self.args.kernel_limit)
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

                w, h = func(filter_coeff, i)
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

    def scipy_freqz_FIR(self, filter_coeff, i):
        '''
        Helper function for the signal.freqz function.
        '''
        sr = self.args.sr_data
        return signal.freqz(b=filter_coeff, fs=sr, worN=5*sr)

    def scipy_freqz_IIR(self, filter_coeff, i):
        '''
        Helper function for the signal.freqz function.
        '''
        sr = self.args.sr_data
        filter_coeff = np.append(-1, filter_coeff)
        return signal.freqz(b=1, a=filter_coeff, fs=sr, worN=5*sr)

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


class ConvAR(WavenetSimple):
    def build_model(self, args):
        ch = args.num_channels
        self.conv = Conv1d(
            ch, ch, kernel_size=args.rf, groups=args.groups)

    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        preds = self.conv(data['inputs'])

        loss = self.criterion(preds, data['targets'][:, :, -preds.shape[2]:])

        losses = {'trainloss/optloss/Training loss: ': loss,
                  'valloss/valcriterion/Validation loss: ': loss,
                  'valloss/saveloss/none': loss}

        return losses


class WavenetSimpleUniToMulti(WavenetSimple):
    '''
    Initialize weights of multivariate model with univariate model,
    and then only train the multivariate weights
    '''
    def loaded_(self, args):
        self.args = args

        # save current weights
        self.fconv = self.first_conv
        self.lconv = self.last_conv
        self.layers = self.cnn_layers

        # create new multivariate model
        self.build_model(args)

        # initialize weights
        self.reset_weights()

    def reset_weights(self):
        '''
        Reset univariate portions of the weights to the original univariates.
        '''
        with torch.torch.no_grad():
            self.unitomulti(self.first_conv, self.fconv)
            self.unitomulti(self.last_conv, self.lconv)
            for new_layer, old_layer in zip(self.cnn_layers, self.layers):
                self.unitomulti(new_layer, old_layer)

    def unitomulti(self, new, old):
        '''
        Replace new univaraite weights with old weights.
        '''
        gin = int(old.in_channels/old.groups)
        gout = int(old.out_channels/old.groups)
        for i in range(old.groups):
            new.weight[i*gout:(i+1)*gout, i*gin:(i+1)*gin, :] = old.weight[i*gout:(i+1)*gout, :, :]

    def loss_(self, x, i=0, sid=None, train=True, criterion=None):
        # reset weights on each iteration
        self.reset_weights()

        return super(WavenetSimpleUniToMulti, self).loss(x, i, sid, train, criterion)


class WavenetSimpleShared(WavenetSimple):
    '''
    WavenetSimple but the same model for each sensor.
    '''
    def __init__(self, args):
        super(WavenetSimpleShared, self).__init__(args)
        ch = self.ch
        self.inp_ch = 1
        self.out_ch = 1
        self.build_model(args)

        self.ch = ch

    def forward(self, x, sid=None):
        bs = x.shape[0]
        chn = x.shape[1]
        x = x.reshape(-1, 1, x.shape[2])
        out, x = super(WavenetSimpleShared, self).forward(x)

        x = x.reshape(bs, chn, -1, x.shape[2])
        # this changed from the correct behaviour for experimentation
        return out.reshape(bs, chn, -1), x.reshape(bs, -1, x.shape[3])


class WavenetSimpleChannelUp(WavenetSimple):
    '''
    WavenetSimple but channel dimension increases with each layer.
    '''
    def build_model(self, args):
        '''
        Specify the layers of the model.
        '''
        modules = []
        self.ch = int(args.ch_mult * self.inp_ch)
        self.first_conv = Conv1d(
            self.inp_ch, self.ch, kernel_size=1, groups=1)

        for i, rate in enumerate(args.dilations):
            modules.append(Conv1d(self.ch,
                                  int(1.5*self.ch),
                                  kernel_size=args.kernel_size,
                                  dilation=rate,
                                  groups=args.groups))
            self.ch = int(1.5*self.ch)

        # 1x1 convolution to go back to original channel dimension
        self.last_conv = Conv1d(
            self.ch, self.out_ch, kernel_size=1, groups=1)

        self.cnn_layers = Sequential(*modules)


class WavenetSimpleSTS(WavenetSimple):
    '''
    Wavenet with 3 consecutive blocks: Spatial -> Temporal -> Spatial
    '''
    def build_model(self, args):
        '''
        Specify the layers of the model.
        '''
        self.ch = args.ch_mult * self.inp_ch

        # spatial beginning
        self.spatial_conv1 = Conv1d(
            self.inp_ch, self.ch, kernel_size=1, groups=1)
        self.spatial_conv2 = Conv1d(
            self.ch, self.ch, kernel_size=1, groups=1)

        # spatial end
        self.spatial_conv3 = Conv1d(
            self.ch, self.ch, kernel_size=1, groups=1)
        self.spatial_conv4 = Conv1d(
            self.ch, self.out_ch, kernel_size=1, groups=1)

        self.inp_ch = 1
        self.out_ch = 1
        super(WavenetSimpleSTS, self).build_model(args)

    def forward(self, x, sid=None):
        bs = x.shape[0]

        # first do 2 spatial convolutions
        x = self.activation(self.dropout(self.spatial_conv1(x)))
        x = self.activation(self.dropout(self.spatial_conv2(x)))
        x = x.reshape(bs, -1, int(self.ch/self.inp_ch), x.shape[2])
        x = x.reshape(-1, x.shape[2], x.shape[3])

        # then do a number of temporal convolutions
        for layer in self.cnn_layers:
            x = self.activation(self.dropout(layer(x)))

        # finish with 2 more spatial convolutions
        x = x.reshape(bs, -1, x.shape[1], x.shape[2])
        x = x.reshape(x.shape[0], -1, x.shape[3])
        x = self.activation(self.dropout(self.spatial_conv3(x)))

        return self.spatial_conv4(x), x

    def forward4(self, x, sid=None):
        '''
        Same as forward,
        but only run up to a specific layer of the temporal part.
        '''
        bs = x.shape[0]
        x = self.activation(self.dropout(self.spatial_conv1(x)))
        x = self.activation(self.dropout(self.spatial_conv2(x)))
        x = x.reshape(bs, -1, int(self.ch/self.inp_ch), x.shape[2])
        x = x.reshape(-1, x.shape[2], x.shape[3])

        # get output of a specific temporal layer based on the receptive field
        lnum = int(np.log(self.args.rf) / np.log(self.args.kernel_size)) - 1
        for i, layer in enumerate(self.cnn_layers):
            x = self.activation(self.dropout(layer(x)))
            if i == lnum:
                break

        x = x.reshape(bs, -1, x.shape[1], x.shape[2])
        x = x.reshape(x.shape[0], -1, x.shape[3])
        return None, x


class WavenetSimplePCA(WavenetSimple):
    '''
    WavenetSimple with a linear transform over channels at beginning and end.
    '''
    def build_model(self, args):
        # wrap Wavenet with a linear transform to reduce channel dimensionality
        self.encoder = Linear(self.inp_ch, args.red_channels, bias=False)
        self.decoder = Linear(args.red_channels, self.inp_ch, bias=False)

        self.out_ch = args.red_channels
        self.inp_ch = args.red_channels
        super(WavenetSimplePCA, self).build_model(args)

    def forward(self, x, sid=None):
        x = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, x = super(WavenetSimplePCA, self).forward(x)
        out = self.decoder(out.permute(0, 2, 1)).permute(0, 2, 1)

        return out, None


class WavenetSimplePCAfixed(WavenetSimplePCA):
    '''
    Same as before, but linear transforms are initialized
    with PCA coefficients and fixed.
    '''
    def build_model(self, args):
        super(WavenetSimplePCAfixed, self).build_model(args)

        pca_model = pickle.load(open(args.pca_path, 'rb'))

        # initialize with PCA coefficients
        enc = torch.Tensor(pca_model.components_)
        dec = torch.Tensor(pca_model.components_.T)
        self.encoder.weight = torch.nn.Parameter(enc)
        self.decoder.weight = torch.nn.Parameter(dec)

        # fix weights
        self.encoder.weight.requires_grad = False
        self.decoder.weight.requires_grad = False


class WavenetSimpleSembConcat(WavenetSimple):
    '''
    Implements simplified wavenet with concatenated subject embeddings.
    '''
    def loaded(self, args):
        super(WavenetSimpleSembConcat, self).loaded(args)
        self.emb_window = False

    def build_model(self, args):
        self.emb_window = False
        self.shuffle_embeddings = False
        self.inp_ch = args.num_channels + args.embedding_dim
        super(WavenetSimpleSembConcat, self).build_model(args)

    def embed(self, x, sid):
        # concatenate subject embeddings with input data
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)

        # shuffle embeddings in a window if needed
        if self.emb_window:
            idx = np.random.rand(*sid[:, :, 0].T.shape).argsort(0)
            a = sid[:, :, 0].T.clone()
            out = a[idx, np.arange(a.shape[1])].T

            w = self.emb_window
            out = out.repeat(w[1] - w[0], 1, 1)
            sid[:, :, w[0]:w[1]] = out.permute(1, 2, 0)

        x = torch.cat((x, sid), dim=1)

        return x

    def get_weights(self, grad=False):
        weights = super(WavenetSimpleSembConcat, self).get_weights(grad)
        if self.args.reg_semb:
            weights.append(self.subject_emb.weight)

        return weights

    def forward(self, x, sid=None):
        if sid is None:
            torch.LongTensor([0]).cuda()

        # shuffle embedding values if needed
        if self.shuffle_embeddings:
            print('This code needs to be checked!')
            subid = int(sid[0].detach().cpu().numpy())
            indices = torch.randperm(self.subject_emb.weight.shape[1])
            w = self.subject_emb.weight.detach()
            w[subid, :] = w[subid, indices]
            self.subject_emb.weight = torch.nn.Parameter(w)

        x = self.embed(x, sid)
        return super(WavenetSimpleSembConcat, self).forward(x)

    def forward4(self, x, sid=None):
        if sid is None:
            torch.LongTensor([0]).cuda()

        x = self.embed(x, sid)
        return super(WavenetSimpleSembConcat, self).forward4(x)

    def layer_output(self, x, num_l, sid=None):
        '''
        Compute the output for a specific layer num_l.
        '''
        if sid is None:
            # repeat x 15 times
            x = x.repeat(self.args.subjects, 1, 1)

            # use all 15 embeddings
            sid = torch.LongTensor(np.arange(self.args.subjects)).cuda()

        x = self.embed(x, sid)
        return super(WavenetSimpleSembConcat, self).layer_output(x, num_l)

    def kernel_network_FIR(self,
                           folder='kernels_network_FIR',
                           generated_data=None):
        '''
        Get FIR properties for each kernel by running the whole network.
        '''
        self.eval()
        name = folder + 'ch' + str(self.args.channel_idx)
        folder = os.path.join(self.args.result_dir, name)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # data is either drawn from gaussian or passed as argument to this func
        shape = (self.args.num_channels, self.args.generate_length)
        data = np.random.normal(0, self.args.generate_noise, shape)
        if generated_data is not None:
            data = generated_data
        data = torch.Tensor(data).cuda().reshape(1, self.args.num_channels, -1)

        # apply subject embedding
        sid = torch.LongTensor([10]).cuda()
        sid = sid.repeat(data.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)
        data = torch.cat((data, sid), dim=1)
        data = self.first_conv(data)

        # loop over whole network
        self.kernel_network_FIR_loop(folder, data)


class WavenetSembDrop(WavenetSimpleSembConcat):
    def embed(self, x, sid):
        # concatenate subject embeddings with input data
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)

        # dropout whole embeddings
        inds = np.arange(sid.shape[0])
        np.random.shuffle(inds)
        inds = inds[:int(len(inds)*self.args.p_drop)]
        sid[inds, :, :] = 0

        x = torch.cat((x, sid), dim=1)

        return x


class WavenetSimpleSembNonlinear1(WavenetSimpleSembConcat):
    def forward(self, x, sid=None):
        '''
        Run a forward pass through the network.
        '''
        x = self.embed(x, sid)

        # only do nonlinear activation after first layer
        x = torch.asinh(self.first_conv(x))

        for layer in self.cnn_layers:
            x = self.activation(self.dropout(layer(x)))

        return self.last_conv(x), x


class WavenetSimpleNonlinearSemb(WavenetSimpleSembConcat):
    def embed(self, x, sid):
        # concatenate subject embeddings with input data
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)

        # make embedding nonlinear
        x = torch.cat((x, torch.asinh(sid)), dim=1)

        return x


class WavenetSimpleChetSemb(WavenetSimple):
    '''
    Add embeddings to input at each layer instead of only the beginning.
    '''
    def build_model(self, args):
        super(WavenetSimpleChetSemb, self).build_model(args)
        emb = args.embedding_dim

        # conditioning layer to be used for subject embeddings
        cond1x1 = [Conv1d(emb, self.ch, kernel_size=1) for _ in args.dilations]
        self.cond1x1 = Sequential(*cond1x1)

    def embed(self, x, sid):
        # concatenate subject embeddings with input data
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)

        return sid

    def forward(self, x, sid):
        x = self.first_conv(x)

        # at each layer use a conditioning layer to add the subject embeddings
        for xlayer, slayer in zip(self.cnn_layers, self.cond1x1):
            x = xlayer(x)
            s = slayer(self.embed(x, sid))
            x = self.activation(self.dropout(x + s))

        return self.last_conv(x), x


class WavenetSimpleSembAdd(WavenetSimple):
    '''
    Implements simplified wavenet with added subject embeddings.
    '''
    def forward(self, x, sid=None):
        # add subject embeddings to input timeseries
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)
        x = x + sid

        return super(WavenetSimpleSembAdd, self).forward(x)


class WavenetSimpleSembMult(WavenetSimple):
    '''
    Implements simplified wavenet with multiplied subject embeddings.
    '''
    def forward(self, x, sid=None):
        # multiply subject embedding with input
        sid = sid.repeat(x.shape[2], 1).permute(1, 0)
        sid = self.subject_emb(sid).permute(0, 2, 1)
        x = x * sid

        return super(WavenetSimpleSembMult, self).forward(x)


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
        modules = [args.pooling(kernel_size=2, stride=2) for r in args.dilations]
        self.maxpool_layers = Sequential(*modules)

    def forward(self, x, sid=None):
        x = self.first_conv(x)

        for conv, pool in zip(self.cnn_layers, self.maxpool_layers):
            x = self.activation(self.dropout(pool(conv(x))))

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
        x = self.run_kernel(x, self.cnn_layers[num_l], num_f)

        return -torch.mean(x)

    def plot_welch(self, x, ax, i):
        '''
        Compute and plot (on ax) welch spectra of x.
        The sampling rate in each layer (i) is halved.
        '''
        sr = self.args.sr_data / 2**i
        super(ConvPoolNet).plot_welch(x, ax, sr)

    def scipy_freqz(self, filter_coeff, i):
        '''
        Helper function for the signal.freqz function. (i: layer index)
        '''
        sr = int(self.args.sr_data / 2**i)
        return signal.freqz(b=filter_coeff, fs=sr, worN=5*sr)


class ConvPoolNetSemb(ConvPoolNet, WavenetSimpleSembConcat):
    def build_model(self, args):
        self.emb_window = False
        self.shuffle_embeddings = False
        self.inp_ch = args.num_channels + args.embedding_dim
        super(ConvPoolNetSemb, self).build_model(args)

    def forward(self, x, sid=None):
        if sid is None:
            torch.LongTensor([0]).cuda()

        x = self.embed(x, sid)
        return super(ConvPoolNetSemb, self).forward(x)

    def forward4(self, x, sid=None):
        if sid is None:
            torch.LongTensor([0]).cuda()

        x = self.embed(x, sid)
        return super(ConvPoolNetSemb, self).forward4(x)


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
