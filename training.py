import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import sails
import torch
import random
import pickle

from scipy import signal
from scipy.io import savemat

from torch.nn import MSELoss
from torch.optim import Adam

from loss import Loss


class Experiment:
    def __init__(self, args):
        '''
        Initialize model, dataset and optimizer.
        '''
        self.args = args
        self.loss = Loss()
        self.val_losses = []
        self.train_losses = []

        if not os.path.isdir(self.args.result_dir):
            os.mkdir(self.args.result_dir)

        # save args
        path = os.path.join(self.args.result_dir, 'args_saved.py')
        os.system('cp ' + args.name + ' ' + path)

        # initialize dataset
        self.dataset = self.args.dataset(self.args)

        # load model if path is specified
        if args.load_model:
            self.model_path = os.path.join(args.load_model, 'model.pt')
            self.model = torch.load(self.model_path)
            #self.args.dataset = self.dataset
            self.model.args = self.args
            #torch.save(self.model, self.model_path, pickle_protocol=4)
        else:
            self.model_path = os.path.join(self.args.result_dir, 'model.pt')
            self.model = self.args.model(self.args).cuda()

        # initialize optimizer
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.args.learning_rate)

        # calculate number of total parameters in model
        parameters = [param.numel() for param in self.model.parameters()]
        print('Number of parameters: ', sum(parameters))

    def train(self):
        '''
        Main training loop over epochs and training batches.
        '''
        best_val = 1000000
        self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()
            self.loss.list = []

            # shuffle each epoch
            # ind = torch.randperm(self.dataset.x_train_t.shape[0])
            # self.dataset.x_train_t = self.dataset.x_train_t[ind, :, :]

            for i in range(self.dataset.train_batches):
                batch, sid = self.dataset.get_train_batch(i)
                if batch.shape[0] < 1:
                    break

                loss, _, _, loss2 = self.model.loss(batch, i, sid, train=True)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.loss.append(loss, loss2)

            if not epoch % self.args.print_freq:
                loss = self.loss.print('Train loss: ')
                self.train_losses.append(loss)

            # only save model if validation loss is best so far
            if not epoch % self.args.val_freq:
                loss = self.evaluate()
                self.val_losses.append(loss)
                if loss < best_val:
                    best_val = loss
                    torch.save(self.model, self.model_path, pickle_protocol=4)

                if self.args.save_curves:
                    self.save_curves()

        self.evaluate()

    def save_curves(self):
        '''
        Save train and validation loss curves to file.
        '''
        val_losses = np.array(self.val_losses)
        val_ratio = int(len(self.train_losses)/len(val_losses))
        val_losses = np.repeat(val_losses, val_ratio)

        plt.semilogy(self.train_losses, linewidth=1, label='training loss')
        plt.semilogy(val_losses, linewidth=1, label='validation loss')
        plt.legend()

        path = os.path.join(self.args.result_dir, 'losses.svg')
        plt.savefig(path, format='svg', dpi=1200)
        plt.close('all')

    def evaluate(self):
        '''
        Evaluate model on the validation dataset.
        '''
        self.loss.list = []
        self.model.eval()

        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss, _, _, loss2 = self.model.loss(batch, i, sid, train=False)
            self.loss.append(loss, loss2)

        loss = self.loss.print('Validation loss: ')
        return loss

    def save_validation(self):
        loss = self.evaluate()

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write(str(loss))

    def save_validation_subs(self):
        '''
        Evaluate model on the validation dataset for each subject.
        '''
        self.model.eval()
        mse = MSELoss(reduction='none').cuda()
        losses = []

        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss, _, _, loss2 = self.model.loss(
                batch, i, sid, train=False, criterion=mse)

            loss = torch.mean(loss, (1, 2)).detach()
            losses.append((sid, loss))

        sid = torch.cat(tuple([loss[0] for loss in losses]))
        loss = torch.cat(tuple([loss[1] for loss in losses]))

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            for i in range(self.args.subjects):
                sub_loss = torch.mean(loss[sid == i]).item()
                f.write(str(sub_loss) + '\n')

    def save_validation_channels(self):
        '''
        Evaluate model for each channel separately
        '''
        self.model.eval()
        mse = MSELoss(reduction='none').cuda()
        losses = []
        outputs = []

        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss, output, _, loss2 = self.model.loss(
                batch, i, sid, train=False, criterion=mse)

            losses.append(torch.mean(loss.detach(), (0, 2)))
            outputs.append(output.detach())

        loss = torch.mean(torch.stack(tuple(losses)), 0)
        one_loss = torch.mean(loss)

        outputs = torch.cat(tuple(outputs)).permute(1, 0, 2)
        outputs = outputs.reshape(outputs.shape[0], -1)
        var = torch.std(outputs, 1)
        one_var = torch.std(torch.flatten(outputs))

        path = os.path.join(self.args.result_dir, 'val_loss_var.txt')
        with open(path, 'w') as f:
            f.write(str(one_loss.item()) + '\t' + str(one_var.item()))

        path = os.path.join(self.args.result_dir, 'val_loss_ch.txt')
        with open(path, 'w') as f:
            for i in range(loss.shape[0]):
                f.write(str(loss[i].item()) + '\t' + str(var[i].item()))
                f.write('\n')

    def pca_sensor_loss(self):
        '''
        Evaluate model for each channel separately
        '''
        self.model.eval()
        self.loss.list = []
        mse = MSELoss().cuda()

        self.args.num_channels = list(range(128))
        self.args.load_data = self.args.load_data2
        self.pca_data = self.args.dataset(self.args)
        outputs = []
        targets = []

        for i in range(self.dataset.val_batches):
            batch, _ = self.dataset.get_val_batch(i)
            batch_pca, _ = self.pca_data.get_val_batch(i)
            loss, output, target, loss2 = self.model.loss(
                batch_pca, i, _, train=False)

            self.loss.append(loss, loss2)

            outputs.append(output.detach())
            targets.append(batch[:, :, -output.shape[2]:])

        self.loss.print('Validation loss: ')

        outputs = torch.cat(tuple(outputs)).permute(0, 2, 1)
        outputs = outputs.reshape(-1, outputs.shape[2]).cpu().numpy()
        outputs = self.dataset.pca_model.inverse_transform(outputs)
        outputs = torch.Tensor(outputs).cuda()

        targets = torch.cat(tuple(targets)).permute(0, 2, 1)
        targets = targets.reshape(-1, outputs.shape[1])

        loss = mse(outputs, targets)
        print(loss.item())

    def repeat_baseline(self):
        '''
        Simple baseline that repeats current timestep as prediction for next.
        '''
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            self.loss.append(self.model.repeat_loss(batch))

        self.loss.print('Repeat baseline loss: ')

    def plot_freqs(self, ax):
        for freq in self.args.freqs:
            ax.axvline(x=freq, color='red')

    def AR_analysis(self, params):
        '''
        Analysie the frequency properties of multivariate AR filters (params)
        '''
        # TODO: this is individual filters but we might also be interested in
        # looking at the sum of filters for a specific output channel
        sr = self.args.sr_data
        filters = params.reshape(params.shape[0], -1).transpose()
        num_filt = min(filters.shape[0], self.args.kernel_limit)
        fig_fir, axs_fir = plt.subplots(num_filt+1, figsize=(15, num_filt*6))
        fig_iir, axs_iir = plt.subplots(num_filt+1, figsize=(15, num_filt*6))

        for i in range(num_filt):
            # frequency spectra as FIR filter
            w, h = signal.freqz(b=filters[i], fs=sr, worN=5*sr)
            axs_fir[i].plot(w, np.abs(h))
            self.plot_freqs(axs_fir[i])

            # frequency spectra as IIR filter
            filter_coeff = np.append(-1, filters[i])
            w, h = signal.freqz(b=1, a=filter_coeff, fs=sr, worN=5*sr)
            axs_iir[i].plot(w, np.abs(h))
            self.plot_freqs(axs_iir[i])

        path = os.path.join(self.args.result_dir, 'AR_FIR.svg')
        fig_fir.savefig(path, format='svg', dpi=1200)
        plt.close('all')
        path = os.path.join(self.args.result_dir, 'AR_IIR.svg')
        fig_iir.savefig(path, format='svg', dpi=1200)
        plt.close('all')

        computation = 'ji,ij->i' if self.args.uni else 'jii,ij->i'
        shape = (self.args.num_channels, self.args.generate_length)
        data = np.random.normal(0, 1, shape)

        # generate in IIR mode
        for t in range(params.shape[0], data.shape[1]):
            inputs = data[:, t-params.shape[0]:t]
            data[:, t] += np.einsum(computation, params, inputs[:, ::-1])

        path = os.path.join(self.args.result_dir, 'generatedAR.mat')
        savemat(path, {'X': data})

    def AR_baseline(self):
        '''
        Train and validate an either uni or multivariate autoregressive model.
        '''
        ts = self.args.timesteps
        self.AR_order = np.arange(self.args.order + 1)

        # prepare data tensors
        x_train = self.dataset.x_train.reshape(self.args.num_channels, -1, 1)
        x_val = self.dataset.x_val
        outputs = np.zeros((x_val.shape[0], x_val.shape[1], ts))
        target = np.zeros((x_val.shape[0], x_val.shape[1], ts))

        func = self.AR_uni if self.args.uni else self.AR_multi
        outputs, target, filters = func(x_train, x_val, outputs, target, ts)

        if self.args.do_anal:
            self.AR_analysis(filters)

        outputs = torch.Tensor(outputs).float().cuda()
        target = torch.Tensor(target).float().cuda()

        # save validation loss and generated variance for all future timesteps
        path = os.path.join(self.args.result_dir, 'timestep_AR.txt')
        with open(path, 'w') as file:
            for i in range(ts):
                loss = self.model.ar_loss(outputs[:, :, i], target[:, :, i])
                var = np.std(outputs[:, :, i].reshape(-1).cpu().numpy())

                file.write(str(loss.item()) + '\t' + str(var) + '\n')
                print('AR validation loss ts', i+1, ': ', loss.item(),
                      ' Variance: ', var)

                path = os.path.join(self.args.result_dir,
                                    'ts' + str(i) + 'AR.txt')
                with open(path, 'w') as f:
                    for ch in range(self.args.num_channels):
                        loss = self.model.ar_loss(outputs[ch, :, i],
                                                  target[ch, :, i])
                        out = outputs[ch, :, i].reshape(-1).cpu().numpy()
                        var = np.std(out)

                        f.write(str(loss.item()) + '\t' + str(var) + '\n')

    def AR_multi(self, x_train, x_val, generated, target, ts, ch='multi'):
        '''
        Train and validate a multivariate AR model.
        '''
        # load or train new model
        path = os.path.join(self.args.AR_load_path, 'ARch' + ch)
        if self.args.save_AR:
            model = sails.modelfit.OLSLinearModel.fit_model(x_train,
                                                            self.AR_order)
            pickle.dump(model, open(path, 'wb'))
        else:
            model = pickle.load(open(path, 'rb'))

        coeff = model.parameters[:, :, 1:]

        # generate prediction for each timestep
        for t in range(model.order, x_val.shape[1] - ts):
            # at each timestep predict in the future recursively up to ts
            for i in range(ts):
                # true input + generated past so far
                concat = (x_val[:, t-model.order+i:t], generated[:, t, :i])
                inputs = np.concatenate(concat, axis=1)[:, ::-1]

                target[:, t, i] = x_val[:, t+i]
                generated[:, t, i] = np.einsum('iij,ij->i', coeff, inputs)

        return generated, target, coeff.transpose(2, 0, 1)

    def AR_uni(self, x_train, x_val, generated, target, ts):
        '''
        Train and validate a univariate AR model.
        '''
        filters = []

        # create a separate AR model for each channel.
        for ch in range(x_val.shape[0]):
            gen, targ, params = self.AR_multi(x_train[ch:ch+1, :, :],
                                              x_val[ch:ch+1, :],
                                              generated[ch:ch+1, :, :],
                                              target[ch:ch+1, :, :],
                                              ts,
                                              str(ch))
            generated[ch:ch+1, :, :] = gen
            target[ch:ch+1, :, :] = targ
            filters.append(params)

        filters = np.concatenate(tuple(filters), axis=1)
        return generated, target, filters.reshape(-1, x_val.shape[0])

    def recursive(self):
        '''
        Evaluate a trained model for recursive multi-step prediction.
        '''
        self.model.eval()
        self.model.timesteps = 1
        ts = self.args.timesteps
        bs = self.args.batch_size
        shift = self.args.rf

        x_val = self.dataset.x_val
        length = x_val.shape[1]-shift-ts
        x_val = np.array([x_val[:, i:i+shift+ts] for i in range(length)])
        x_val = torch.Tensor(x_val).float().cuda()

        # create arrays for each future timestep
        generated = torch.zeros((x_val.shape[0], x_val.shape[1], ts)).cuda()
        target = torch.zeros((x_val.shape[0], x_val.shape[1], ts)).cuda()

        # loop over batches and future timesteps
        for b in range(int(x_val.shape[0]/bs)+1):
            for i in range(ts):
                # concatenate past timesteps and predictions by the model
                past = x_val[b*bs:(b+1)*bs, :, i:shift]
                predicted = generated[b*bs:(b+1)*bs, :, :i]
                inputs = torch.cat((past, predicted), axis=2)

                output = self.model(inputs)[0].reshape(inputs.shape[0], -1)
                generated[b*bs:(b+1)*bs, :, i] = output.detach()
                target[b*bs:(b+1)*bs, :, i] = x_val[b*bs:(b+1)*bs, :, shift+i]

        # compute loss for each future timestep separately
        file = open(os.path.join(self.args.result_dir, 'timesteps.txt'), 'w')
        for i in range(ts):
            loss = self.model.ar_loss(generated[:, :, i], target[:, :, i])
            var = np.std(generated[:, :, i].reshape(-1).cpu().numpy())
            print('WN validation loss ts', i+1, ': ', loss.item(),
                  ' Variance: ', var)

            file.write(str(loss.item()) + '\t' + str(var) + '\n')

            # self.freq_loss(generated, target)

        file.close()

    def feature_importance(self):
        '''
        Evaluate how important is each timestep to prediction
        '''
        self.model.eval()
        rf = self.args.rf
        bs = self.args.batch_size
        chs = list(range(self.args.num_channels))

        x_val = self.dataset.x_val
        x_val = torch.Tensor(x_val).float().cuda()

        losses = []
        # loop over timestep positions in input
        for i in list(range(1, rf))[::-1]:
            losses.append([])
            # loop over timesteps
            for b in range(int((x_val.shape[1]-rf)/bs)-1):
                inputs = [x_val[:, b*bs+k:b*bs+k+rf+1] for k in range(bs)]
                inputs = torch.stack(inputs)

                # permute the i-th timestep
                random.shuffle(chs)
                inputs = (inputs[:, :, :i],
                          inputs[:, chs, i:i+1],
                          inputs[:, :, i+1:])

                inputs = torch.cat(inputs, dim=2)

                loss = self.model.loss(inputs, b, train=False)[0]
                losses[-1].append(loss.item())

            losses[-1] = sum(losses[-1])/len(losses[-1])
            print(losses[-1])

        # save to file loss for each permuted timestep
        path = os.path.join(self.args.result_dir, 'permutation_losses.txt')
        with open(path, 'w') as f:
            for loss in losses:
                f.write(str(loss) + '\n')

    def freq_loss(self, generated, target):
        '''
        Compute loss for each event type in the data.
        generated: generated timeseries for validation data
        target: ground truth validation data
        '''
        length = self.dataset.x_val.shape[1]
        for i, freq in enumerate(self.args.freqs):
            part = self.dataset.stc[self.shift+self.ts:length]
            ind = np.where(part == i)
            loss = self.model.ar_loss(generated[ind, :, 0], target[ind, :, 0])
            print('Frequency ', freq, ' loss: ', loss.item())

    def noise_mse(self):
        '''
        Compute the MSE of two normal distributions.
        '''
        data = torch.normal(0, 1, size=(1, self.args.sr_data*50)).cuda()
        target = torch.normal(0, 1, size=(1, self.args.sr_data*50)).cuda()
        criterion = MSELoss().cuda()
        loss = criterion(data, target)
        print(loss.item())

    def analyse_activations(self):
        self.model.analyse_activations()

    def analyse_kernels(self):
        self.model.analyse_kernels()

    def plot_kernels(self):
        self.model.plot_kernels()

    def generate(self):
        self.model.generate()

    def kernel_network_FIR(self):
        self.model.kernel_network_FIR()

    def kernel_network_IIR(self):
        self.model.kernel_network_IIR()

    def save_embeddings(self):
        if self.args.subjects > 0:
            self.model.save_embeddings()

    def input_freq(self):
        figsize = (15, 10*self.args.num_channels)
        fig, axs = plt.subplots(self.args.num_channels+1, figsize=figsize)

        x_train = self.dataset.x_train

        # loop over all channels
        for ch in range(x_train.shape[0]):
            # compute fft of learned input
            self.model.plot_welch(x_train[ch], axs[ch])

        filename = os.path.join(self.args.result_dir, 'input_freq.svg')
        fig.savefig(filename, format='svg', dpi=2400)
        plt.close('all')


def main(Args):
    args = Args()
    # if list of paths is given, then process everything individually
    for i, d_path in enumerate(args.data_path):
        args_pass = Args()
        args_pass.data_path = d_path
        args_pass.result_dir = args.result_dir[i]
        args_pass.dump_data = args.dump_data[i]
        args_pass.load_data = args.load_data[i]
        args_pass.norm_path = args.norm_path[i]
        args_pass.pca_path = args.pca_path[i]
        args_pass.AR_load_path = args.AR_load_path[i]
        args_pass.load_model = args.load_model[i]

        # skip if subject does not exist
        if not (os.path.isfile(d_path) or os.path.isdir(d_path)):
            print('Skipping ' + d_path)
            continue

        e = Experiment(args_pass)

        if Args.func['repeat_baseline']:
            e.input_freq()
            e.repeat_baseline()
        if Args.func['AR_baseline']:
            e.AR_baseline()
        if Args.func['train']:
            e.train()
        if Args.func['analyse_kernels']:
            e.analyse_kernels()
        if Args.func['plot_kernels']:
            e.plot_kernels()
        if Args.func['generate']:
            e.generate()
        if Args.func['recursive']:
            e.recursive()
        if Args.func['kernel_network_FIR']:
            e.kernel_network_FIR()
        if Args.func['kernel_network_IIR']:
            e.kernel_network_IIR()
        if Args.func['feature_importance']:
            e.feature_importance()
        if Args.func['save_validation_subs']:
            e.save_validation_subs()
        if Args.func['save_validation_ch']:
            e.save_validation_channels()
        if Args.func['pca_sensor_loss']:
            e.pca_sensor_loss()

        e.save_embeddings()


'''
if __name__ == "__main__":
    main()
'''
