import os
import matplotlib.pyplot as plt
import numpy as np
import sails
import torch

from scipy import signal
from scipy.io import savemat

from torch.nn import MSELoss
from torch.optim import Adam


from args import Args
from loss import Loss

os.environ["NVIDIA_VISIBLE_DEVICES"] = Args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = Args.gpu


class Experiment:
    def __init__(self):
        '''
        Initialize model, dataset and optimizer.
        '''
        self.args = Args()
        self.loss = Loss()
        self.val_losses = []
        self.train_losses = []

        if not os.path.isdir(self.args.result_dir):
            os.mkdir(self.args.result_dir)

        # save args
        path = os.path.join(self.args.result_dir, 'args_saved.py')
        os.system('cp args.py ' + path)

        # initialize dataset
        self.dataset = self.args.dataset(self.args)
        self.args.dataset = self.dataset

        # load model if path is specified
        self.model_path = os.path.join(self.args.result_dir, 'model.pt')
        if self.args.load_model:
            self.model = torch.load(self.model_path)
            self.model.args = self.args
        else:
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

        for epoch in range(self.args.epochs):
            self.model.train()
            self.loss.list = []

            # shuffle each epoch
            # ind = torch.randperm(self.dataset.x_train_t.shape[0])
            # self.dataset.x_train_t = self.dataset.x_train_t[ind, :, :]

            for i in range(self.dataset.train_batches):
                batch = self.dataset.get_train_batch(i)
                loss, _, _, loss2 = self.model.loss(batch, i, train=True)

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
                    torch.save(self.model, self.model_path)

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
            batch = self.dataset.get_val_batch(i)
            loss, _, _, loss2 = self.model.loss(batch, i, train=False)
            self.loss.append(loss, loss2)

        loss = self.loss.print('Validation loss: ')
        return loss

    def repeat_baseline(self):
        '''
        Simple baseline that repeats current timestep as prediction for next.
        '''
        for i in range(self.dataset.val_batches):
            batch = self.dataset.get_val_batch(i)
            self.loss.append(self.model.repeat_loss(batch))

        self.loss.print('Repeat baseline loss: ')

    def AR_filter_analysis(self, params):
        '''
        Analysie the frequency properties of the learned AR filters.
        params: the AR coefficients
        '''
        # TODO: this is individual filters but we might also be interested in
        # looking at the sum of filters for a specific output channel
        sr = self.args.sr_data
        filters = params.reshape(-1, params.shape[2])
        num_filt = min(filters.shape[0], 300)
        fig_fir, axs_fir = plt.subplots(num_filt+1, figsize=(20, num_filt*3))
        fig_iir, axs_iir = plt.subplots(num_filt+1, figsize=(20, num_filt*3))

        for i in range(num_filt):
            # frequency spectra as FIR filter
            w, h = signal.freqz(b=filters[i], fs=sr, worN=5*sr)
            axs_fir[i].plot(w, np.abs(h))

            # frequency spectra as IIR filter
            filter_coeff = np.append(1, filters[i])
            w, h = signal.freqz(b=1, a=filter_coeff, fs=sr, worN=5*sr)
            axs_fir[i].plot(w, np.abs(h))

        path = os.path.join(self.args.result_dir, 'AR_FIR.svg')
        fig_fir.savefig(path, format='svg', dpi=1200)
        plt.close('all')
        path = os.path.join(self.args.result_dir, 'AR_IIR.svg')
        fig_iir.savefig(path, format='svg', dpi=1200)
        plt.close('all')

        # generate in IIR mode
        data = np.random.normal(0, 1, (self.args.sr_data*1000))
        for t in range(params.shape[2], len(data)):
            # TODO: fix and validate this part
            data[t] += params * data[t-params.shape[2]:t]

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

        self.AR_filter_analysis(filters)

        outputs = torch.Tensor(outputs).float().cuda()
        target = torch.Tensor(target).float().cuda()

        # save validation loss and generated variance for all future timesteps
        path = os.path.join(self.args.result_dir, 'timestep_AR.txt')
        with open(path, 'w') as f:
            for i in range(ts):
                loss = self.model.ar_loss(outputs[:, :, i], target[:, :, i])
                var = np.std(outputs[:, :, i].reshape(-1).cpu().numpy())

                f.write(str(loss.item()) + '\t' + str(var) + '\n')
                print('AR validation loss ts', i+1, ': ', loss.item(),
                      ' Variance: ', var)

    def AR_multi(self, x_train, x_val, generated, target, ts):
        '''
        Train and validate a multivariate AR model.
        '''
        model = sails.modelfit.OLSLinearModel.fit_model(x_train, self.AR_order)

        # generate prediction for each timestep
        for t in range(model.order, x_val.shape[1] - ts):
            # at each timestep predict in the future recursively up to ts
            for i in range(ts):
                concat = (x_val[:, t-model.order+i:t], generated[:, t, :i])
                inputs = np.concatenate(concat, axis=1)

                target[:, t, i] = x_val[:, t+i]
                # TODO: fix and validate this part
                generated[:, t, i] = model.parameters * inputs

        return generated, target, model.parameters[:, :, 1:]

    def AR_uni(self, x_train, x_val, generated, target, ts):
        '''
        Train and validate a univariate AR model.
        '''
        filters = []

        # create a separate AR model for each channel.
        for ch in range(x_val.shape[0]):
            gen, targ, params = self.AR_multi(x_train[ch:ch+1, :, :],
                                              x_val[ch:ch+1, :, :],
                                              generated[ch:ch+1, :, :],
                                              target[ch:ch+1, :, :],
                                              ts)
            generated[ch:ch+1, :, :] = gen
            target[ch:ch+1, :, :] = targ
            filters.append(params)

        # TODO: which axis to concatenate over
        filters = np.concatenate(*filters, axis=5)
        return generated, target, filters

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

            #self.freq_loss(generated, target)

        file.close()

    def freq_loss(self, generated, target):
        '''
        Compute loss for each event type in the data.
        generated: generated timeseries for validation data
        target: ground truth validation data
        '''
        for i, freq in enumerate(self.args.freqs):
            part = self.dataset.stc[self.shift+self.ts:self.dataset.x_val.shape[1]]
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


def main():
    e = Experiment()

    if Args.func['repeat_baseline']:
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


if __name__ == "__main__":
    main()
