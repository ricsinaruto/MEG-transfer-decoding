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
        Initialize model and dataset using an Args object.
        '''
        self.args = args
        self.loss = Loss()
        self.val_losses = []
        self.train_losses = []

        # create folder for results
        if os.path.isdir(self.args.result_dir):
            print('Result already directory exists, writing to it.')
            print(self.args.result_dir)
        else:
            os.mkdir(self.args.result_dir)
            print('New result directory created.')
            print(self.args.result_dir)

        # save args object
        path = os.path.join(self.args.result_dir, 'args_saved.py')
        os.system('cp ' + args.name + ' ' + path)

        # initialize dataset
        if args.load_dataset:
            self.dataset = args.dataset(args)
            print('Dataset initialized.')

        # load model if path is specified
        if args.load_model:
            self.model_path = os.path.join(args.load_model, 'model.pt')
            self.model = torch.load(self.model_path)
            self.model_path = os.path.join(self.args.result_dir, 'model.pt')
            self.model.loaded(args)
            self.model.cuda()
            print('Model loaded from file.')
            #self.args.dataset = self.dataset
        else:
            self.model_path = os.path.join(self.args.result_dir, 'model.pt')
            try:
                self.model = self.args.model(self.args).cuda()
                print('Model initialized with cuda.')
            except:  # if cuda not available or not cuda model
                self.model = self.args.model(self.args)
                print('Model initialized without cuda.')

        try:
            # calculate number of total parameters in model
            parameters = [param.numel() for param in self.model.parameters()]
            print('Number of parameters: ', sum(parameters), flush=True)
        except:
            print('Can\'t calculate number of parameters.')

    def train(self):
        '''
        Main training loop over epochs and training batches.
        '''
        # initialize optimizer
        optimizer = Adam(self.model.parameters(),
                         lr=self.args.learning_rate,
                         weight_decay=self.args.alpha_norm)

        # start with a pass over the validation set
        best_val = 1000000
        self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()
            self.loss.dict = {}

            # save initial model
            if epoch == 0:
                path = os.path.join(self.args.result_dir, 'model_init.pt')
                torch.save(self.model, path, pickle_protocol=4)
                print('Model saved to result directory.')

            # loop over batches
            for i in range(self.dataset.train_batches):
                batch, sid = self.dataset.get_train_batch(i)
                # need to check whether it's an empty batch
                try:
                    if batch.shape[0] < 1:
                        break
                except AttributeError:
                    pass

                losses, _, _, = self.model.loss(batch, i, sid, train=True)

                # optimize model according to the optimization loss
                optkey = [key for key in losses if 'optloss' in key]
                losses[optkey[0]].backward()
                optimizer.step()
                optimizer.zero_grad()
                self.loss.append(losses)

            # print training losses
            if not epoch % self.args.print_freq:
                losses = self.loss.print('trainloss')
                self.train_losses.append([losses[k] for k in losses])

            # run validation pass and save model
            if not epoch % self.args.val_freq:
                losses, _, _ = self.evaluate()
                loss = [losses[k] for k in losses if 'saveloss' in k]
                losses = [losses[k] for k in losses if 'saveloss' not in k]

                self.val_losses.append(losses)

                # only save model if validation loss is best so far
                if loss[0] < best_val:
                    best_val = loss[0]
                    torch.save(self.model, self.model_path, pickle_protocol=4)
                    print('Validation loss improved, model saved.')

                # save loss plots if needed
                if self.args.save_curves:
                    self.save_curves()

        # wrap up training and save validation loss
        self.model.end()
        self.save_validation()

    def save_curves(self):
        '''
        Save train and validation loss plots to file.
        '''
        val_losses = np.array(self.val_losses)
        train_losses = np.array(self.train_losses)

        if val_losses.shape[0] > 2:
            val_ratio = int((train_losses.shape[0]-1)/(val_losses.shape[0]-1))
            val_losses = np.repeat(val_losses, val_ratio, axis=0)

            plt.semilogy(train_losses, linewidth=1, label='training losses')
            plt.semilogy(val_losses, linewidth=1, label='validation losses')
            plt.legend()

            path = os.path.join(self.args.result_dir, 'losses.svg')
            plt.savefig(path, format='svg', dpi=1200)
            plt.close('all')

    def evaluate(self):
        '''
        Evaluate model on the validation dataset.
        '''
        self.loss.dict = {}
        self.model.eval()

        # loop over validation batches
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss, output, target = self.model.loss(batch, i, sid, train=False)
            self.loss.append(loss)

        losses = self.loss.print('valloss')
        return losses, output, target

    def save_validation(self):
        '''
        Save validation loss to file.
        '''
        loss, output, target = self.evaluate()

        # print variance if needed
        if output is not None and target is not None:
            print(torch.std((output-target).flatten()))

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write(str(loss))

    def save_validation_subs(self):
        '''
        Print validation losses separately on each subject's dataset.
        '''
        self.model.eval()
        losses = []

        # don't reduce the loss so we can separate it according to subjects
        mse = MSELoss(reduction='none').cuda()

        # loop over validation batches
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss_dict, _, _ = self.model.loss(
                batch, i, sid, train=False, criterion=mse)

            loss = [loss_dict[k] for k in loss_dict if 'valcriterion' in k]
            #loss = torch.mean(loss, (1, 2)).detach()
            loss = loss[0].detach()
            losses.append((sid, loss))

        sid = torch.cat(tuple([loss[0] for loss in losses]))
        loss = torch.cat(tuple([loss[1] for loss in losses]))

        path = os.path.join(self.args.result_dir, 'val_loss_subs.txt')
        with open(path, 'w') as f:
            for i in range(self.args.subjects):
                sub_loss = torch.mean(loss[sid == i]).item()
                f.write(str(sub_loss) + '\n')

    def save_validation_channels(self):
        '''
        Evaluate model for each channel separately.
        Needs an update to work.
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
        Loss between pca and non-pca data.
        Needs an update to work.
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

    def lda_baseline(self):
        '''
        Train a separate linear model across time windows.
        '''
        hw = self.args.halfwin
        times = self.dataset.x_val_t.shape[2]
        accs = []

        for i in range(hw, times-hw):
            # select input slice
            x_t = self.dataset.x_train_t.clone()
            x_v = self.dataset.x_val_t.clone()

            # train model on a specific time window
            acc = self.model.run(x_t, x_v, (i-hw, i+hw))
            print(acc)
            accs.append(str(acc))

            # save each model
            with open(self.model_path + str(i), 'wb') as file:
                pickle.dump(self.model, file)

            # re-initialize model
            self.model.init_model()

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(accs))

        return accs

    def lda_pairwise(self):
        '''
        Train LDA for pairwise classification.
        '''
        accuracies = []
        nc = self.args.num_classes
        chn = self.args.num_channels - 1
        x_t = self.dataset.x_train_t.clone()
        x_v = self.dataset.x_val_t.clone()

        # do a first pass to fit PCA
        self.lda_baseline()

        for c1 in range(nc):
            for c2 in range(c1+1, nc):
                # set labels for pairwise classification
                self.dataset.x_train_t = x_t.clone()
                self.dataset.x_train_t[x_t[:, chn, 0] == c1, chn, :] = 0
                self.dataset.x_train_t[x_t[:, chn, 0] == c2, chn, :] = 1

                # select trials from these 2 classes
                inds = x_t[:, chn, 0] == c1 or x_t[:, chn, 0] == c2
                self.dataset.x_train_t = self.dataset.x_train_t[inds, :, :]

                # repeat for validation data
                self.dataset.x_val_t = x_v.clone()
                self.dataset.x_val_t[x_v[:, chn, 0] == c1, chn, :] = 0
                self.dataset.x_val_t[x_v[:, chn, 0] == c2, chn, :] = 1

                inds = x_v[:, chn, 0] == c1 or x_v[:, chn, 0] == c2
                self.dataset.x_val_t = self.dataset.x_val_t[inds, :, :]

                accs = self.lda_baseline()
                accuracies.append(accs)

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(accuracies))

    def lda_eval(self):
        '''
        Evaluate any linear classifier on each subject separately.
        '''
        # load model
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

        path = os.path.join(self.args.result_dir, 'val_loss_subs.txt')
        with open(path, 'w') as f:
            for i in range(self.args.subjects):
                inds = self.dataset.sub_id['val'] == i
                x_val = self.dataset.x_val_t[inds, :, :]

                acc = self.model.eval(x_val)
                f.write(str(acc) + '\n')

    def repeat_baseline(self):
        '''
        Simple baseline that repeats current timestep as prediction for next.
        '''
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            self.loss.append(self.model.repeat_loss(batch))

        self.loss.print('valloss')

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
        Evaluate how important is each timestep to prediction.
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

    def PFIemb(self):
        '''
        Permutation Feature Importance (PFI) function for subject embeddings.
        '''
        loss_list = []
        hw = self.args.halfwin
        times = self.dataset.x_val_t.shape[2]

        # slide over the epoch and always permute embeddings within a window
        for i in range(hw-1, times-hw):
            if i > hw-1:
                self.model.wavenet.emb_window = (i-hw, i+hw)

            losses, _, _ = self.evaluate()
            loss = [losses[k] for k in losses if 'Validation accuracy' in k]
            loss_list.append(str(loss[0]))

        name = 'val_loss_PFIemb' + str(hw) + '.txt'
        path = os.path.join(self.args.result_dir, name)
        with open(path, 'w') as f:
            f.write('\n'.join(loss_list))

    def PFIts(self):
        '''
        Newer Permutation Feature Importance (PFI) function for timesteps.
        Could also try the inverse, like a combination of PFI and window_eval.
        '''
        loss_list = []
        hw = self.args.halfwin
        val_t = self.dataset.x_val_t.clone()
        shuffled_val_t = self.dataset.x_val_t.clone()
        chn = val_t.shape[1] - 1
        times = val_t.shape[2]

        # first permute channels across all timesteps
        idx = np.random.rand(*val_t[:, :chn, 0].T.shape).argsort(0)
        for i in range(times):
            a = shuffled_val_t[:, :chn, i].T
            out = a[idx, np.arange(a.shape[1])].T
            shuffled_val_t[:, :chn, i] = out

        # slide over the epoch and always permute timesteps within a window
        for i in range(hw, times-hw):
            self.dataset.x_val_t = val_t.clone()
            if i > 0:
                # either permute inside or outside the window
                if self.args.PFI_inverse:
                    window = shuffled_val_t[:, :chn, :i-hw].clone()
                    self.dataset.x_val_t[:, :chn, :i-hw] = window
                    window = shuffled_val_t[:, :chn, i+hw:].clone()
                    self.dataset.x_val_t[:, :chn, i+hw:] = window
                else:
                    window = shuffled_val_t[:, :chn, i-hw:i+hw].clone()
                    self.dataset.x_val_t[:, :chn, i-hw:i+hw] = window

            losses, _, _ = self.evaluate()
            loss = [losses[k] for k in losses if 'Validation accuracy' in k]
            loss_list.append(str(loss[0]))

        path = os.path.join(self.args.result_dir, 'val_loss_PFIts.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(loss_list))

    def PFIch(self):
        '''
        Newer Permutation Feature Importance (PFI) function for channels.
        '''
        loss_list = []
        top_chs = self.args.closest_chs
        val_t = self.dataset.x_val_t.clone()
        shuffled_val_t = self.dataset.x_val_t.clone()
        chn = val_t.shape[1] - 1
        tmin = self.args.pfich_timesteps[0]
        tmax = self.args.pfich_timesteps[1]

        # read a file containing closest channels to each channel location
        path = os.path.join(self.args.result_dir, 'closest' + str(top_chs))
        with open(path, 'rb') as f:
            closest_k = pickle.load(f)

        # first permute timesteps across all channels
        idx = np.random.rand(*val_t[:, 0, :].T.shape).argsort(0)
        for i in range(chn):
            a = shuffled_val_t[:, i, :].T
            out = a[idx, np.arange(a.shape[1])].T
            shuffled_val_t[:, i, :] = out

        # evaluate without channel shuffling
        loss, _, _ = self.evaluate()
        loss = [loss[k] for k in loss if 'Validation acc' in k]
        loss_list.append(str(loss[0]))

        # loop over channels and permute channels in vicinity of i-th channel
        for i in range(int(chn/3)):
            self.dataset.x_val_t = val_t.clone()

            # need to select mag and 2 grads
            a = np.array(closest_k[i]) * 3
            chn_idx = np.append(np.append(a, a+1), a+2)

            # shuffle closest k channels
            if self.args.PFI_inverse:
                mask = np.ones(chn+1, np.bool)
                mask[chn_idx] = 0
                mask[chn] = 0
                window = shuffled_val_t[:, mask, tmin:tmax].clone()
                self.dataset.x_val_t[:, mask, tmin:tmax] = window
            else:
                window = shuffled_val_t[:, chn_idx, tmin:tmax].clone()
                self.dataset.x_val_t[:, chn_idx, tmin:tmax] = window

            loss, _, _ = self.evaluate()
            loss = [loss[k] for k in loss if 'Validation acc' in k]
            loss_list.append(str(loss[0]))

        # save accuracies to file
        name = 'val_loss_PFIch' + str(top_chs) + '.txt'
        path = os.path.join(self.args.result_dir, name)
        with open(path, 'w') as f:
            f.write('\n'.join(loss_list))

        '''
        # loop over permutations of other channels than i
        # this is to improve noisy results when permuting 1 channel
        chn_idx = np.array([c for c in range(chn) if c != i])
        chn_idx = np.random.choice(chn_idx, size=19, replace=False)
        chn_idx = np.append(chn_idx, i)
        '''

    def freq_loss(self, generated, target):
        '''
        Compute loss for each event type in the simulated data.
        generated: generated timeseries for validation data
        target: ground truth validation data
        '''
        length = self.dataset.x_val.shape[1]
        for i, freq in enumerate(self.args.freqs):
            part = self.dataset.stc[self.shift+self.ts:length]
            ind = np.where(part == i)
            loss = self.model.ar_loss(generated[ind, :, 0], target[ind, :, 0])
            print('Frequency ', freq, ' loss: ', loss.item())

    def input_freq(self):
        '''
        Plot the PSD of the training data.
        '''
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

    def test(self):
        '''
        Run model with shuffled embeddings.
        '''
        self.model.eval()
        loss_list = []

        self.model.wavenet.shuffle_embeddings = False
        for i in range(101):
            if i == 1:
                self.model.wavenet.shuffle_embeddings = True

            batch, sid = self.dataset.get_val_batch(0)
            loss, output, target = self.model.loss(batch, 0, sid, train=False)
            loss = [loss[k] for k in loss if 'Validation accuracy' in k]

            loss = str(loss[0].item())
            loss_list.append(loss)
            print(loss)

        path = os.path.join(self.args.result_dir, 'val_loss_embshuffle.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(loss_list))

    def window_eval(self):
        '''
        Slide a window over validation data and zero out values outside of it.
        '''
        loss_list = []
        inp_halfwin = 128
        inp_win = 256
        halfwin = self.args.halfwin
        sig_len = 256
        val_t = self.dataset.x_val_t[:, :, -275:-19]

        # slide window over timesteps
        for i in range(halfwin, sig_len - halfwin):
            start = 0 if i < inp_halfwin else i-inp_halfwin
            start = start if i + inp_halfwin < sig_len else sig_len - inp_win
            self.dataset.x_val_t = val_t[:, :, start:start+inp_win].clone()

            # values outside the window are set to 0
            self.dataset.x_val_t[:, :306, :i-start-halfwin] = 0
            self.dataset.x_val_t[:, :306, i-start+halfwin:] = 0
            print(self.dataset.x_val_t[0, 0, :])

            losses, _, _ = self.evaluate()
            loss = [losses[k] for k in losses if 'Validation accuracy' in k]
            loss_list.append(str(loss[0]))

        path = os.path.join(
            self.args.result_dir, 'val_loss_windows' + str(halfwin) + '.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(loss_list))

    def plot_freqs(self, ax):
        for freq in self.args.freqs:
            ax.axvline(x=freq, color='red')

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

    def compare_layers(self):
        self.model.compare_layers()

    def save_embeddings(self):
        # only run this if there are multiple subjects
        if self.args.subjects > 0:
            self.model.save_embeddings()


def main(Args):
    '''
    Main function creating an experiment object and running everything.
    This should be called from launch.py, and it needs an Args object.
    '''
    args = Args()

    def checklist(x, i):
        # check if an argument is list or not
        return x[i] if isinstance(x, list) else x

    # if list of paths is given, then process everything individually
    for i, d_path in enumerate(args.data_path):
        args_pass = Args()
        args_pass.data_path = d_path
        args_pass.result_dir = checklist(args.result_dir, i)
        args_pass.dump_data = checklist(args.dump_data, i)
        args_pass.load_data = checklist(args.load_data, i)
        args_pass.norm_path = checklist(args.norm_path, i)
        args_pass.pca_path = checklist(args.pca_path, i)
        args_pass.AR_load_path = checklist(args.AR_load_path, i)
        args_pass.load_model = checklist(args.load_model, i)
        args_pass.max_trials = checklist(args.max_trials, i)
        args_pass.batch_size = checklist(args.batch_size, i)
        args_pass.p_drop = checklist(args.p_drop, i)
        args_pass.learning_rate = checklist(args.learning_rate, i)
        args_pass.load_conv = checklist(args.load_conv, i)
        args_pass.compare_model = checklist(args.compare_model, i)

        # skip if subject does not exist
        if not (os.path.isfile(d_path) or os.path.isdir(d_path)):
            print('Skipping ' + d_path, flush=True)
            continue

        e = Experiment(args_pass)

        # only run the functions specified in args
        if Args.func['repeat_baseline']:
            e.input_freq()
            e.repeat_baseline()
        if Args.func['AR_baseline']:
            e.AR_baseline()
        if Args.func['LDA_baseline']:
            e.lda_baseline()
        if Args.func['LDA_pairwise']:
            e.lda_pairwise()
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
        if Args.func['compare_layers']:
            e.compare_layers()
        if Args.func['test']:
            e.test()
        if Args.func['LDA_eval']:
            e.lda_eval()
        if Args.func['window_eval']:
            e.window_eval()
        if Args.func['PFIts']:
            e.PFIts()
        if Args.func['PFIch']:
            e.PFIch()
        if Args.func['PFIemb']:
            e.PFIemb()

        e.save_embeddings()
