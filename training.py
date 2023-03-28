import os
import matplotlib.pyplot as plt
import numpy as np
import sails
import torch
import random
import pickle
import traceback
from copy import deepcopy

from scipy import signal
from scipy.io import savemat
from scipy.fft import fft, ifft, rfft, irfft

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from torch.nn import MSELoss
from torch.optim import Adam

from loss import Loss
from classifiers_linear import LDA, LDA_average_trials


class Experiment:
    def __init__(self, args, dataset=None):
        '''
        Initialize model and dataset using an Args object.
        '''
        self.args = args
        self.loss = Loss()
        self.val_losses = []
        self.train_losses = []

        if args.fix_seed:
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

        # create folder for results
        if os.path.isdir(self.args.result_dir):
            print('Result directory already exists, writing to it.',
                  flush=True)
            print(self.args.result_dir, flush=True)
        else:
            os.makedirs(self.args.result_dir)
            print('New result directory created.', flush=True)
            print(self.args.result_dir, flush=True)

        # save args object
        path = os.path.join(self.args.result_dir, 'args_saved.py')
        os.system('cp ' + args.name + ' ' + path)

        # initialize dataset
        if dataset is not None:
            self.dataset = dataset
        elif args.load_dataset:
            self.dataset = args.dataset(args)
            print('Dataset initialized.', flush=True)

        # load model if path is specified
        if args.load_model:
            if 'model' in args.load_model:
                self.model_path = args.load_model
            else:
                self.model_path = os.path.join(args.load_model, 'model.pt')

            # LDA vs deep learning models
            try:
                self.model = torch.load(self.model_path)
                self.model.loaded(args)
                self.model.cuda()
            except:
                self.model = pickle.load(open(self.model_path, 'rb'))
                self.model.loaded(args)

            self.model_path = os.path.join(self.args.result_dir, 'model.pt')

            print('Model loaded from file.', flush=True)
            #self.args.dataset = self.dataset
        else:
            self.model_path = os.path.join(self.args.result_dir, 'model.pt')
            self.model = args.model(args)

            try:
                self.model = self.model.cuda()
                print('Model initialized with cuda.', flush=True)
            except:  # if cuda not available or not cuda model
                print('Model initialized without cuda.')

        try:
            # calculate number of total parameters in model
            parameters = [param.numel() for param in self.model.parameters()]
            print('Number of parameters: ', sum(parameters), flush=True)
        except:
            print('Can\'t calculate number of parameters.', flush=True)

    def train(self):
        '''
        Main training loop over epochs and training batches.
        '''
        # initialize optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Adam(params,
                         lr=self.args.learning_rate,
                         weight_decay=self.args.alpha_norm)

        # use cosine annealing
        if self.args.anneal_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.dataset.train_batches)

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
                print('Model saved to result directory.', flush=True)

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
                losses = self.loss.append(losses)

                if self.args.anneal_lr:
                    scheduler.step()

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
                    print('Validation loss improved, model saved.', flush=True)

                    # also compute test loss
                    self.testing()

                else:
                    path = self.model_path.strip('.pt') + '_epoch.pt'
                    torch.save(self.model, path, pickle_protocol=4)

                # save loss plots if needed
                if self.args.save_curves:
                    self.save_curves()

        # wrap up training, save model and validation loss
        if self.args.epochs:
            path = self.model_path.strip('.pt') + '_end.pt'
            torch.save(self.model, path, pickle_protocol=4)
        self.model.end()
        self.save_validation()

    def testing(self):
        '''
        Evaluate model on the test set.
        '''
        self.loss.dict = {}
        self.model.eval()

        # loop over test batches
        for i in range(self.dataset.test_batches):
            batch, sid = self.dataset.get_test_batch(i)
            loss, output, target = self.model.loss(batch, i, sid, train=False)
            self.loss.append(loss)

        losses = self.loss.print('valloss')

        path = os.path.join(self.args.result_dir, 'test_loss.txt')
        with open(path, 'w') as f:
            f.write(str(losses))

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
        outputs = []
        targets = []

        # loop over validation batches
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            loss, output, target = self.model.loss(batch, i, sid, train=False)
            self.loss.append(loss)

            #outputs.append(output)
            #targets.append(target)

        # concatenate self.model.losses
        #loss = torch.concat((self.model.losses), dim=0).numpy()

        # save self.model.losses to file
        #path = os.path.join(self.args.result_dir, 'losses.npy')
        #np.save(path, loss)

        losses = self.loss.print('valloss')
        #return losses, torch.cat(outputs), torch.cat(targets)
        return losses, None, None

    def classify(self):
        self.model.eval()

        accs = []

        batches, _ = self.dataset.get_val_batch(0)

        batch = {'inputs': [], 'targets': [], 'condition': [], 'sid': []}
        # loop over validation batches
        for i in range(0, batches['sid'].shape[0], self.args.batch_size):
            
            for k in batch.keys():
                batch[k] = batches[k][i:i+self.args.batch_size]

            acc = self.model.classify(batch)
            accs.append(acc.double())

        accs = torch.cat(accs, dim=0)
        accs = accs.mean(dim=0)
        print(accs)

    def evaluate_train(self):
        '''
        Evaluate model on the validation dataset.
        '''
        self.loss.dict = {}
        self.model.eval()

        # loop over validation batches
        for i in range(self.dataset.train_batches):
            batch, sid = self.dataset.get_train_batch(i)
            loss, output, target = self.model.loss(batch, i, sid, train=False)
            self.loss.append(loss)

        losses = self.loss.print('valloss')

    def save_validation(self):
        '''
        Save validation loss to file.
        '''
        loss, output, target = self.evaluate()

        # print variance if needed
        #if output is not None and target is not None:
        #    print(torch.std((output-target).flatten()))

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write(str(loss))

        self.testing()

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

    def save_validation_timepoints(self):
        '''
        Print validation losses separately on each subject's dataset.
        '''
        self.model.eval()
        accs = []

        # loop over validation batches
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            _, _, acc = self.model.loss(batch, i, sid, train=False)

            accs.append((batch[:, -2, 0].cpu(), acc.detach().cpu()))

        sid = torch.cat(tuple([loss[0] for loss in accs]))
        loss = torch.cat(tuple([loss[1] for loss in accs]))

        path = os.path.join(self.args.result_dir, 'val_loss_timepoints.txt')
        with open(path, 'w') as f:
            for i in range(self.args.sample_rate):
                time_loss = torch.mean(loss[sid == i]).item()
                f.write(str(time_loss) + '\n')

    def confusion_matrix(self):
        '''
        Compute confusion matrix of model predictions and save to file.
        '''
        _, outputs, targets = self.evaluate()

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        mat = confusion_matrix(targets, outputs)
        path = os.path.join(self.args.result_dir, 'confusion_matrix')
        np.save(path, mat)

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

    def lda_baseline(self, filewrite=True, run_test=True, save_model=True, reinit=True):
        '''
        Train a separate linear model across time windows.
        '''
        hw = self.args.halfwin
        times = self.dataset.x_test_t.shape[2]
        train_accs = []
        val_accs = []
        test_accs = []

        print('Val data shape: ', self.dataset.x_val_t.shape)

        if isinstance(self.model, LDA_average_trials):
            times = times//4

        for i in range(hw, times-hw+1):
            # select input slice
            x_t = self.dataset.x_train_t.clone()
            x_v = self.dataset.x_val_t.clone()

            end = hw-1 if self.args.halfwin_uneven else hw

            # train model on a specific time window
            acc, _, _ = self.model.run(x_t,
                                       x_v,
                                       (i-hw, i+end),
                                       sid_train=self.dataset.sub_id['train'],
                                       sid_val=self.dataset.sub_id['val'])
            print(acc)
            val_accs.append(str(acc))

            if run_test:
                acc, _, _ = self.model.eval(
                    x_t, (i-hw, i+end), sid=self.dataset.sub_id['train'])
                train_accs.append(str(acc))

                x_t = self.dataset.x_test_t.clone()
                acc, _, _ = self.model.eval(
                    x_t, (i-hw, i+end), sid=self.dataset.sub_id['test'])
                test_accs.append(str(acc))

            if save_model:
                # save each model
                with open(self.model_path + str(i), 'wb') as file:
                    pickle.dump(self.model, file)

            # re-initialize model
            if reinit:
                self.model.init_model()

        if filewrite:
            path = os.path.join(self.args.result_dir, 'val_loss.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(val_accs))

            path = os.path.join(self.args.result_dir, 'train_loss.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(train_accs))

            path = os.path.join(self.args.result_dir, 'test_loss.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(test_accs))

        return val_accs

    def lda_crossval(self):
        '''
        Loop over subjects in dataset and train lda model in cross-validation.
        '''

        # copy train and val data
        x_train = self.dataset.x_train_t.clone()
        train_sid = self.dataset.sub_id['train'].clone()

        for i in range(self.args.subjects):
            # select i-th subject
            self.dataset.x_train_t = x_train[train_sid != i].clone()
            self.dataset.x_val_t = x_train[train_sid == i].clone()
            self.dataset.x_test_t = self.dataset.x_val_t

            self.dataset.sub_id['train'] = train_sid[train_sid != i].clone()

            # set correct number of train and val batches
            bs = self.args.batch_size
            self.dataset.bs['train'] = self.dataset.find_bs(
                bs, self.dataset.x_train_t.shape[0])
            self.dataset.bs['val'] = self.dataset.find_bs(
                bs, self.dataset.x_val_t.shape[0])
            self.dataset.bs['test'] = self.dataset.find_bs(
                bs, self.dataset.x_test_t.shape[0])

            self.dataset.train_batches = int(
                self.dataset.x_train_t.shape[0] / self.dataset.bs['train'])
            self.dataset.val_batches = int(
                self.dataset.x_val_t.shape[0] / self.dataset.bs['val'])
            self.dataset.test_batches = int(
                self.dataset.x_test_t.shape[0] / self.dataset.bs['test'])

            # train model
            if isinstance(self.model, LDA):
                self.lda_baseline(
                    filewrite=False, run_test=False, save_model=False)
            else:
                self.train()

    def lda_crossval_pairs(self):
        '''
        Loop over subjects in dataset and train lda model in cross-validation.
        '''

        # copy train and val data
        x_train = self.dataset.x_train_t.clone()

        accs = np.zeros((self.args.subjects, self.args.subjects))
        for i in range(self.args.subjects):
            # select i-th subject
            self.dataset.x_train_t = x_train[self.dataset.sub_id['train'] == i].clone()
            self.dataset.x_val_t = x_train[self.dataset.sub_id['train'] == i].clone()

            # train model
            self.lda_baseline(filewrite=False, run_test=False, save_model=False, reinit=False)

            # evaluate model on all other subjects
            for j in range(self.args.subjects):
                x_val = x_train[self.dataset.sub_id['train'] == j].clone()
                acc, _, _ = self.model.eval(x_val)
                accs[i, j] = acc

        # save accs
        path = os.path.join(self.args.result_dir, 'accs.npy')
        np.save(path, accs)

    def lda_channel(self):
        '''
        Train a separate lda_baseline model for each channel.
        '''
        # copy self.datasset.x
        x_train = self.dataset.x_train_t.clone()
        x_val = self.dataset.x_val_t.clone()
        x_test = self.dataset.x_test_t.clone()

        accs = []
        num_ch = self.args.num_channels
        for ch in range(int(num_ch/3)):
            current_chs = [ch*3, ch*3+1, ch*3+2, num_ch]
            self.dataset.x_train_t = x_train[:, current_chs, :]
            self.dataset.x_val_t = x_val[:, current_chs, :]
            self.dataset.x_test_t = x_test[:, current_chs, :]
            self.args.num_channels = 3

            acc = self.lda_baseline()
            accs.append(acc[0])

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(accs))

    def lda_pairwise(self):
        '''
        Train LDA for pairwise classification.
        '''
        accuracies = []
        nc = self.args.num_classes
        chn = self.args.num_channels
        x_t = self.dataset.x_train_t.clone()
        x_v = self.dataset.x_val_t.clone()

        # do a first pass to fit PCA
        self.lda_baseline()

        for c1 in range(nc):
            for c2 in range(c1+1, nc):
                # set labels for pairwise classification
                c1_inds = x_t[:, chn, 0] == c1
                c2_inds = x_t[:, chn, 0] == c2

                self.dataset.x_train_t = x_t.clone()
                self.dataset.x_train_t[c1_inds, chn, :] = 0
                self.dataset.x_train_t[c2_inds, chn, :] = 1

                # select trials from these 2 classes
                inds = (c1_inds) | (c2_inds)
                #print(x_t[:100, chn, 0])
                self.dataset.x_train_t = self.dataset.x_train_t[inds, :, :]
                #print(self.dataset.x_train_t.shape)

                # repeat for validation data
                c1_inds = x_v[:, chn, 0] == c1
                c2_inds = x_v[:, chn, 0] == c2

                self.dataset.x_val_t = x_v.clone()
                self.dataset.x_val_t[c1_inds, chn, :] = 0
                self.dataset.x_val_t[c2_inds, chn, :] = 1

                inds = (c1_inds) | (c2_inds)
                self.dataset.x_val_t = self.dataset.x_val_t[inds, :, :]

                accs = self.lda_baseline(filewrite=False,
                                         run_test=False,
                                         save_model=False)
                accuracies.append(';'.join(accs))

        path = os.path.join(self.args.result_dir, 'val_loss.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(accuracies))

    def lda_eval(self):
        '''
        Evaluate any linear classifier on each subject separately.
        '''
        path = os.path.join(self.args.result_dir, 'val_loss_subs.txt')
        with open(path, 'w') as f:
            for i in range(self.args.subjects):
                inds = self.dataset.sub_id['val'] == i
                x_val = self.dataset.x_val_t[inds, :, :]

                acc, _, _ = self.model.eval(x_val)
                print(acc)
                f.write(str(acc) + '\n')

    def lda_eval_train_subs(self):
        '''
        Evaluate any linear classifier on each subject separately.
        '''
        path = os.path.join(self.args.result_dir, 'train_loss_subs.txt')
        with open(path, 'w') as f:
            for i in range(self.args.subjects):
                inds = self.dataset.sub_id['train'] == i
                x_val = self.dataset.x_train_t[inds, :, :]

                acc, _, _ = self.model.eval(x_val)
                print(acc)
                f.write(str(acc) + '\n')

    def lda_eval_train(self):
        '''
        Evaluate any linear classifier on the train data of another dataset.
        '''
        self.model.loaded(self.args)
        path = os.path.join(self.args.result_dir, 'train.txt')
        with open(path, 'w') as f:
            x_train = self.dataset.x_train_t

            acc, _, _ = self.model.eval(x_train, sid=self.dataset.sub_id['train'])
            print(acc)
            f.write(str(acc) + '\n')

    def lda_eval_train_ensemble(self):
        '''
        Run an ensemble of individual classifiers on the train data of another dataset.
        '''

        # load models
        models = []
        for path in self.args.load_models:
            with open(path, 'rb') as file:
                model = pickle.load(file)
                model.loaded(self.args)
                models.append(model)
        
        path = os.path.join(self.args.result_dir, 'left_out_train.txt')
        with open(path, 'w') as f:
            x_train = self.dataset.x_train_t
            
            class_probs = []
            for model in models:
                class_prob, y_val = model.predict(x_train)
                class_prob = np.argmax(class_prob, axis=1)
                class_probs.append(class_prob)

            class_probs = np.array(class_probs)
            
            # majority vote
            y_preds = []
            for i in range(class_probs.shape[1]):
                y_pred = np.argmax(np.bincount(class_probs[:, i]))
                y_preds.append(y_pred)

            '''
            # aggregate probabilities
            class_probs = np.mean(class_probs, axis=0)
            y_preds = np.argmax(class_probs, axis=1)
            '''

            y_preds = np.array(y_preds)
            acc = np.mean(y_preds == y_val)

            print(acc)
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
        x_train = self.dataset.x_train[:self.args.num_channels, :]
        x_train = x_train.reshape(self.args.num_channels, -1, 1)
        x_val = self.dataset.x_val[:self.args.num_channels, :]

        outputs = np.zeros((x_val.shape[0], x_val.shape[1], ts))
        target = np.zeros((x_val.shape[0], x_val.shape[1], ts))

        func = self.AR_uni if self.args.uni else self.AR_multi
        outputs, target, filters = func(x_train, x_val, outputs, target, ts)

        if self.args.do_anal:
            self.AR_analysis(filters)

        # save outputs and targets as numpy arrays
        path = os.path.join(self.args.result_dir, 'outputs.npy')
        np.save(path, outputs)
        path = os.path.join(self.args.result_dir, 'target.npy')
        np.save(path, target)

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
        Only used for forecasting models, not classification.
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

    def evaluate_(self, data, og=None):
        '''
        Evaluation helper function for PFI.
        '''
        losses, _, _ = self.evaluate()
        loss = [losses[k] for k in losses if 'Validation accuracy' in k]
        return loss[0]

    def LDA_eval_(self, data, og=None):
        '''
        Evaluation helper for PFI for linear models.
        '''
        acc, _, _ = self.model.eval(data)
        return acc

    def kernelPFI(self, data, og=False):
        '''
        Helper function for PFI to compute kernel output deviations.
        '''
        self.model.eval()
        ch = self.args.num_channels
        num_l = len(self.args.dilations)

        # get output at specific kernels
        outputs_batch = []
        for i in range(self.dataset.val_batches):
            batch, sid = self.dataset.get_val_batch(i)
            out = self.model.kernelPFI(batch[:, :ch, :], sid)
            outputs_batch.append(out)

        # concatenate along trial dimension
        outputs = []
        for i in range(len(outputs_batch[0])):
            out = torch.cat([o[i] for o in outputs_batch])
            outputs.append(out)

        if og:
            # set original kernel outputs
            self.kernelPFI_outputs = outputs
            ret = np.zeros(self.args.kernel_limit*num_l)
        else:
            # compute kernel output deviation
            ret = []
            for og, new in zip(self.kernelPFI_outputs, outputs):
                ret.append(torch.linalg.norm(og-new).numpy())
            ret = np.array(ret)

        return ret

    def PFIts(self):
        '''
        Permutation Feature Importance (PFI) function for timesteps.
        '''
        dataset = self.dataset.x_val_t
        if not self.args.PFI_val:
            dataset = self.dataset.x_train_t

        hw = self.args.halfwin
        val_t = dataset.clone()
        shuffled_val_t = dataset.clone()
        chn = val_t.shape[1] - 1
        times = val_t.shape[2]

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # evaluate without channel shuffling
        og_loss = val_func(dataset, True)

        perm_list = []
        for p in range(self.args.PFI_perms):
            # first permute channels across all timesteps
            idx = np.random.rand(*val_t[:, :chn, 0].T.shape).argsort(0)
            for i in range(times):
                a = shuffled_val_t[:, :chn, i].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_t[:, :chn, i] = out

            loss_list = [og_loss]
            # slide over the epoch and always permute timesteps within a window
            for i in range(hw, times-hw, self.args.PFI_step):
                self.dataset.x_val_t = val_t.clone()
                if i > hw:
                    # either permute inside or outside the window
                    if self.args.PFI_inverse:
                        window = shuffled_val_t[:, :chn, :i-hw].clone()
                        self.dataset.x_val_t[:, :chn, :i-hw] = window
                        window = shuffled_val_t[:, :chn, i+hw:].clone()
                        self.dataset.x_val_t[:, :chn, i+hw:] = window
                    else:
                        window = shuffled_val_t[:, :chn, i-hw:i+hw].clone()
                        self.dataset.x_val_t[:, :chn, i-hw:i+hw] = window

                loss = val_func(self.dataset.x_val_t)
                loss_list.append(loss)

            perm_list.append(np.array(loss_list))

        # save accuracies to file
        path = os.path.join(self.args.result_dir, 'val_loss_PFIts.npy')
        np.save(path, np.array(perm_list))

    def PFIfreq(self):
        '''
        Permutation Feature Importance (PFI) function for frequencies.
        '''
        hw = self.args.halfwin
        chn = self.dataset.x_val_t.shape[1] - 1
        times = self.dataset.x_val_t.shape[2]

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # compute fft
        val_fft = rfft(self.dataset.x_val_t[:, :chn, :].cpu().numpy())
        shuffled_val_fft = val_fft.copy()
        #samples = [val_fft[0, 0, :].copy()]

        # original loss without shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        perm_list = []
        for p in range(self.args.PFI_perms):
            id_arr = np.arange(val_fft.shape[0])
            # shuffle frequency components across channels
            idx = np.random.rand(*val_fft[:, :, 0].T.shape).argsort(0)
            for i in range(times//2+1):
                a = shuffled_val_fft[:, :, i].T
                out = a[idx, id_arr].T
                shuffled_val_fft[:, :, i] = out

            loss_list = [og_loss]
            # slide over epoch and always permute frequencies within a window
            for i in range(hw, times//2-hw):
                dataset_val_fft = val_fft.copy()
                win = shuffled_val_fft[:, :, i-hw:i+hw+1].copy()
                dataset_val_fft[:, :, i-hw:i+hw+1] = win
                #dataset_val_fft[:, :, i-hw:i+hw+1] = 0 + 0j

                #samples.append(dataset_val_fft[0, 0, :].copy())

                # inverse fourier transform
                data = torch.Tensor(irfft(dataset_val_fft))
                self.dataset.x_val_t[:, :chn, :] = data

                loss = val_func(self.dataset.x_val_t)
                loss_list.append(loss)

            perm_list.append(np.array(loss_list))

        # save accuracies to file
        path = os.path.join(self.args.result_dir,
                            'val_loss_PFIfreqs' + str(hw*2) + '.npy')
        np.save(path, np.array(perm_list))

        #np.save(path + '_samples', np.array(samples))

    def PFIfreq_ch(self):
        '''
        Permutation Feature Importance (PFI) function for frequencies.
        spectral PFI is done separately for each channel.
        '''
        top_chs = self.args.closest_chs
        hw = self.args.halfwin
        chn = self.dataset.x_val_t.shape[1] - 1
        times = self.dataset.x_val_t.shape[2]

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # read a file containing closest channels to each channel location
        with open(top_chs, 'rb') as f:
            closest_k = pickle.load(f)

        # evaluate without channel shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        # compute fft
        val_fft = rfft(self.dataset.x_val_t[:, :chn, :].cpu().numpy())
        shuffled_val_fft = val_fft.copy()

        samples = [val_fft[0, 0, :].copy()]

        perm_list = []
        for p in range(self.args.PFI_perms):
            # shuffle frequency components across channels
            idx = np.random.rand(*val_fft[:, :, 0].T.shape).argsort(0)
            for i in range(times//2+1):
                a = shuffled_val_fft[:, :, i].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_fft[:, :, i] = out

            windows = []
            # slide over epoch and always permute frequencies within a window
            for i in range(hw, times//2-hw):

                loss_list = [og_loss]
                for c in range(int(chn/3)):
                    # need to select magnetometer and 2 gradiometers
                    a = np.array(closest_k[c]) * 3
                    chn_idx = np.append(np.append(a, a+1), a+2)

                    dataset_val_fft = val_fft.copy()
                    win1 = shuffled_val_fft[:, chn_idx, i-hw:i+hw+1].copy()
                    dataset_val_fft[:, chn_idx, i-hw:i+hw+1] = win1
                    #samples.append(dataset_val_fft[0, 0, :].copy())

                    # inverse fourier transform
                    data = torch.Tensor(irfft(dataset_val_fft))
                    self.dataset.x_val_t[:, :chn, :] = data

                    loss = val_func(self.dataset.x_val_t)
                    loss_list.append(loss)

                windows.append(np.array(loss_list))

            perm_list.append(np.array(windows))

        # save accuracies to file
        path = os.path.join(self.args.result_dir,
                            'val_loss_PFIfreqs_ch' + str(hw*2) + '.npy')
        np.save(path, np.array(perm_list))

        np.save(path + '_samples', np.array(samples))

    def PFIfreq_ts_(self):
        '''
        Permutation Feature Importance (PFI) function for frequencies.
        spectral PFI is done separately for each channel.
        '''
        hw = self.args.halfwin
        ts_hw = self.args.pfich_timesteps
        chn = self.dataset.x_val_t.shape[1] - 1
        times = self.dataset.x_val_t.shape[2]

        val_t = self.dataset.x_val_t.clone()

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # evaluate without channel shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        # compute fft
        val_fft = rfft(self.dataset.x_val_t[:, :chn, :].cpu().numpy())
        shuffled_val_fft = val_fft.copy()

        perm_list = []
        for p in range(self.args.PFI_perms):
            # shuffle frequency components across channels
            idx = np.random.rand(*val_fft[:, :, 0].T.shape).argsort(0)
            for i in range(times//2+1):
                a = shuffled_val_fft[:, :, i].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_fft[:, :, i] = out

            windows = []
            # slide over epoch and always permute frequencies within a window
            for i in range(hw, times//2-hw):
                dataset_val_fft = val_fft.copy()
                win1 = shuffled_val_fft[:, :, i-hw:i+hw+1].copy()
                dataset_val_fft[:, :, i-hw:i+hw+1] = win1

                # inverse fourier transform
                data = torch.Tensor(irfft(dataset_val_fft))

                loss_list = [og_loss]
                for t in range(ts_hw, times-ts_hw, self.args.PFI_step):
                    self.dataset.x_val_t = val_t.clone()

                    window = data[:, :chn, t-ts_hw:t+ts_hw].clone()
                    self.dataset.x_val_t[:, :chn, t-ts_hw:t+ts_hw] = window

                    loss = val_func(self.dataset.x_val_t)
                    loss_list.append(loss)

                windows.append(np.array(loss_list))

            perm_list.append(np.array(windows))

        # save accuracies to file
        path = os.path.join(self.args.result_dir,
                            f'val_loss_PFIfreqs{hw*2}_ts{ts_hw*2}.npy')
        np.save(path, np.array(perm_list))

    def PFIfreq_ts(self):
        '''
        Permutation Feature Importance (PFI) function for frequencies.
        spectral PFI is done separately for each channel.
        '''
        hw = self.args.halfwin
        ts_hw = self.args.pfich_timesteps
        chn = self.dataset.x_val_t.shape[1] - 1
        sr = self.args.sr_data

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # evaluate without channel shuffling
        og_loss = val_func(self.dataset.x_val_t, True)

        # compute fft
        data = self.dataset.x_val_t[:, :chn, :].cpu().numpy()
        freqs, times, val_fft = signal.stft(data,
                                            fs=sr,
                                            window='hamming',
                                            nperseg=ts_hw*2,
                                            noverlap=ts_hw*2-1,
                                            boundary=None)
        val_fft = val_fft.copy()
        shuffled_val_fft = val_fft.copy()

        perm_list = []
        for p in range(self.args.PFI_perms):
            id_arr = np.arange(val_fft.shape[0])
            idx = np.random.rand(*val_fft[:, :, 0, 0].T.shape).argsort(0)
            for f in range(len(freqs)):
                for t in range(0, len(times), self.args.PFI_step):
                    a = shuffled_val_fft[:, :, f, t].T
                    out = a[idx, id_arr].T
                    shuffled_val_fft[:, :, f, t] = out

            windows = []
            # slide over epoch and always permute frequencies within a window
            for f in range(len(freqs)):

                loss_list = [og_loss]
                for t in range(hw, len(times)-hw, self.args.PFI_step):
                    dataset_val_fft = val_fft.copy()

                    win = shuffled_val_fft[:, :, f:f+1, t-hw:t+hw+1].copy()
                    dataset_val_fft[:, :, f:f+1, t-hw:t+hw+1] = win

                    # inverse fourier transform
                    _, X = signal.istft(dataset_val_fft,
                                        fs=sr,
                                        window='hamming',
                                        nperseg=ts_hw*2,
                                        noverlap=ts_hw*2-1,
                                        boundary=None)
                    self.dataset.x_val_t[:, :chn, :] = torch.Tensor(X)

                    loss = val_func(self.dataset.x_val_t)
                    loss_list.append(loss)

                windows.append(np.array(loss_list))

            perm_list.append(np.array(windows))

        # save accuracies to file
        path = os.path.join(self.args.result_dir,
                            f'val_loss_PFIfreqs_ts{hw*2}.npy')
        np.save(path, np.array(perm_list))

    def PFIch(self):
        '''
        Permutation Feature Importance (PFI) function for channels.
        '''
        # opm indices
        opm_inds = []
        for i in range(51):
            if i < 7:
                opm_inds.append(np.array([i*3, i*3+1, i*3+2]))
            elif i == 7:
                opm_inds.append(np.array([i*3, i*3+1]))
            else:
                opm_inds.append(np.array([i*3-1, i*3, i*3+1]))

        multi = self.args.chn_multi
        dataset = self.dataset.x_val_t
        if not self.args.PFI_val:
            dataset = self.dataset.x_train_t

        top_chs = self.args.closest_chs
        val_t = dataset.clone()
        shuffled_val_t = dataset.clone()
        chn = val_t.shape[1] - 1

        # whether dealing with LDA or deep learning models
        lda_or_not = isinstance(self.model, LDA)
        val_func = self.LDA_eval_ if lda_or_not else self.evaluate_
        if self.args.kernelPFI:
            val_func = self.kernelPFI

        # read a file containing closest channels to each channel location
        with open(top_chs, 'rb') as f:
            closest_k = pickle.load(f)

        # evaluate without channel shuffling
        og_loss = val_func(dataset, True)

        # loop over channels and permute channels in vicinity of i-th channel
        perm_list = []
        for p in range(self.args.PFI_perms):
            # first permute timesteps across all channels
            idx = np.random.rand(*val_t[:, 0, :].T.shape).argsort(0)
            for i in range(chn):
                a = shuffled_val_t[:, i, :].T
                out = a[idx, np.arange(a.shape[1])].T
                shuffled_val_t[:, i, :] = out

            windows = []
            # loop over time windows for spatiotemporal PFI
            for t in self.args.pfich_timesteps:
                tmin = t[0]
                tmax = t[1]
                loss_list = [og_loss]

                for i in range(int(chn/multi)):
                    dataset = val_t.clone()

                    # need to select magnetometer and 2 gradiometers
                    a = np.array(closest_k[i]) * multi

                    if multi > 1:
                        chn_idx = np.append(np.append(a, a+1), a+2)
                    else:
                        chn_idx = a

                    if 'opm' in self.args.data_path:
                        chn_idx = opm_inds[i]

                    # shuffle closest k channels
                    if self.args.PFI_inverse:
                        mask = np.ones(chn+1, np.bool)
                        mask[chn_idx] = 0
                        mask[chn] = 0
                        window = shuffled_val_t[:, mask, tmin:tmax].clone()
                        dataset[:, mask, tmin:tmax] = window
                    else:
                        window = shuffled_val_t[:, chn_idx, tmin:tmax].clone()
                        dataset[:, chn_idx, tmin:tmax] = window

                    loss = val_func(dataset)
                    loss_list.append(loss)

                windows.append(np.array(loss_list))

            perm_list.append(np.array(windows))

        # save accuracies to file
        fname = os.path.basename(os.path.normpath(top_chs))
        name = 'val_loss_PFI' + fname + '.npy'
        path = os.path.join(self.args.result_dir, name)
        np.save(path, np.array(perm_list))

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

    def multi2pair(self):
        '''
        Get pairwise accuracies from multiclass model
        '''
        hw = self.args.halfwin
        nc = self.args.num_classes
        times = self.dataset.x_val_t.shape[2]
        accs = []

        for i in range(hw, times-hw+1):
            # load correct model
            model_path = os.path.join(
                '/'.join(self.model_path.split('/')[:-1]), 'model.pt' + str(i))

            if 'model' in self.args.load_model:
                model_path = self.args.load_model
            self.model = pickle.load(open(model_path, 'rb'))
            self.model.loaded(self.args)

            # select input slice
            x_t = self.dataset.x_train_t.clone()
            x_v = self.dataset.x_val_t.clone()
            end = hw-1 if self.args.halfwin_uneven else hw

            # train model on a specific time window
            probs, targets = self.model.predict(x_v, (i-hw, i+end), x_t)

            pairwise = []
            for c1 in range(nc):
                for c2 in range(c1+1, nc):
                    inds = (targets == c1) | (targets == c2)

                    choice = probs[inds, c1] > probs[inds, c2]
                    predicted = [c1 if p else c2 for p in choice]
                    acc = accuracy_score(np.array(predicted), targets[inds])
                    pairwise.append(acc)

            accs.append(str(np.mean(np.array(pairwise))))

            if 'forest' in self.args.result_dir:
                with open(model_path, 'wb') as file:
                    pickle.dump(self.model, file)

        path = os.path.join(self.args.result_dir, 'pairwise.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(accs))

        return accs

    def gradient_analysis(self):
        self.model.gradient_analysis(self.args)

    def model_inversion(self):
        '''
        Haufe model inversion method.
        '''
        def diag_block_mat(L):
            shp = L[0].shape
            N = len(L)
            r = range(N)
            out = np.zeros((N, shp[0], N, shp[1]))
            out[r, :, r, :] = L
            return out.reshape(np.asarray(shp)*N)

        self.model.lda_norm = False

        x_train = self.dataset.x_train_t[:, :-1, :2*self.args.halfwin-1]
        x_train = x_train.permute(0, 2, 1).cpu().numpy()

        x_val = self.dataset.x_val_t[:, :-1, :2*self.args.halfwin-1]
        x_val = x_val.permute(0, 2, 1).cpu().numpy()
        ts = x_val.shape[1]

        # get data covariance
        #x_val = x_val.reshape(-1, x_val.shape[2]) - self.model.pca.mean_
        #x_val = x_val.reshape(-1, ts, x_val.shape[1])
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_val_cov = np.cov(x_val.T)

        x_train = x_train.reshape(x_train.shape[0], -1)
        x_train_cov = np.cov(x_train.T)

        # apply pca
        '''
        pca = PCA(x_train.shape[1])
        x_train = x_train.transpose(0, 2, 1)
        x_train = x_train.reshape(-1, x_train.shape[2])
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        '''

        # pca into matrix form
        #pca_w = self.model.pca.components_.T
        #pca_w = diag_block_mat([pca_w] * ts)

        #print(self.model.model.intercept_)

        weights = self.model.model.coef_.T
        # transform conv_weights from 306x80 to 306*50 x 80*50
        '''
        conv_w = self.model.spatial_conv.weight.detach().cpu().numpy()
        conv_w = conv_w.squeeze().T
        conv_w = diag_block_mat([conv_w] * ts)
        '''
        '''
        dump_data = '/'.join(self.args.dump_data.split('/')[:-1])
        dump_data = '_'.join(dump_data.split('_')[:-1]) + '_pca306_nonorm/c'
        self.args.dump_data = dump_data
        dataset = self.args.dataset(self.args)
        '''

        # get latent factors covariance
        w = (0, 2*self.args.halfwin)
        #data, target, outputs = self.model.get_output(self.dataset.x_val_t, w)

        # compare outputs
        #outputs2 = np.einsum('bm, mc, ck -> bk',
        #                     x_val[:, :], pca_w, weights)
        outputs2 = x_train @ weights
        output_cov = np.linalg.inv(np.cov(outputs2.T))

        #print(outputs[0, :100])
        #print(outputs2[0, :100])

        print(x_train_cov.shape)
        #print(pca_w.shape)
        print(weights.shape)
        print(output_cov.shape)

        patterns = np.einsum('mm, mk, kk -> mk',
                             x_train_cov, weights, output_cov)

        '''
        # simplified
        data = self.dataset.x_val_t.reshape(data.shape[0], -1).cpu().numpy()
        patterns = []
        for i in range(data.shape[1]):
            patterns.append(np.cov(data[:, i], target))
        '''

        path = os.path.join(self.args.result_dir, 'patterns.npy')
        np.save(path, np.array(patterns))

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
        self.model.generate(self.dataset.x_train_t)

    def kernel_network_FIR(self):
        self.model.kernel_network_FIR()

    def kernel_network_IIR(self):
        self.model.kernel_network_IIR()

    def compare_layers(self):
        self.model.compare_layers()

    def save_embeddings(self):
        # only run this if there are multiple subjects
        if self.args.subjects > 0 and self.args.embedding_dim > 0:
            self.model.save_embeddings()


def main(Args):
    '''
    Main function creating an experiment object and running everything.
    This should be called from launch.py, and it needs an Args object.
    '''
    args = Args()
    dataset = None

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
        args_pass.p_drop = checklist(args.p_drop, i)
        args_pass.load_conv = checklist(args.load_conv, i)
        args_pass.compare_model = checklist(args.compare_model, i)
        args_pass.stft_freq = checklist(args.stft_freq, i)
        args_pass.save_whiten = checklist(args.save_whiten, i)
        args_pass.generate_length = checklist(args.generate_length, i)
        args_pass.sr_data = checklist(args.sr_data, i)
        args_pass.generate_noise = checklist(args.generate_noise, i)

        if isinstance(args.num_channels[0], list):
            args_pass.num_channels = args.num_channels[i]
        if isinstance(args.subjects_data, list):
            if isinstance(args.subjects_data[0], list):
                args_pass.subjects_data = args.subjects_data[i]

        # skip if subject does not exist
        if not isinstance(d_path, list):
            if not (os.path.isfile(d_path) or os.path.isdir(d_path)):
                print('Skipping ' + d_path, flush=True)
                continue

        # separate loops for cross-validation (handled by args.split)
        num_loops = len(args.split) if isinstance(args.split, list) else 1
        split_len = num_loops

        # separate loops for different training set ratios
        if isinstance(args.max_trials, list):
            num_loops = len(args.max_trials)

        # if only using one dataset across loops, initialze it once
        if args.common_dataset and dataset is None:
            dataset = args_pass.dataset(args_pass)
            args_data = deepcopy(args_pass)

        # inner loops, see above
        for n in range(num_loops):
            args_new = deepcopy(args_pass)

            args_new.max_trials = checklist(args_new.max_trials, n)
            args_new.learning_rate = checklist(args_new.learning_rate, n)
            args_new.batch_size = checklist(args_new.batch_size, n)

            # load learned dimensionality reduction for linear models
            if args_new.load_conv:
                if '.pt' not in args_new.load_conv:
                    args_new.load_conv = os.path.join(
                        args_new.load_conv, 'cv' + str(n), 'model.pt')

            # load cross-validation folds accordingly
            if split_len > 1:
                args_new.split = checklist(args.split, n)
                args_new.dump_data = os.path.join(
                    args_new.dump_data + str(n), 'c')
                if args_new.load_data:
                    args_new.load_data = args_new.dump_data
                args_new.result_dir = os.path.join(
                    args_pass.result_dir, 'cv' + str(n))
                if args_new.load_model:
                    paths = args_pass.load_model.split('/')
                    args_new.load_model = os.path.join(
                        '/'.join(paths[:-1]), 'cv' + str(n), paths[-1])

                paths = args_new.load_conv.split('/')
                args_new.load_conv = os.path.join(
                    '/'.join(paths[:-1]), 'cv' + str(n), paths[-1])

            # else use max_trials for looping logic
            elif isinstance(args.max_trials, list):
                name = 'train' + str(args_new.max_trials)
                args_new.result_dir = os.path.join(args_pass.result_dir, name)

            # set common dataset if given
            if args.common_dataset:
                args_data.load_model = args.load_model[i]
                args_data.result_dir = args.result_dir[i]
                e = Experiment(args_data, dataset)
            else:
                e = Experiment(args_new)

            # only run the functions specified in args
            if Args.func.get('repeat_baseline'):
                e.input_freq()
                e.repeat_baseline()
            if Args.func.get('AR_baseline'):
                e.AR_baseline()
            if Args.func.get('LDA_baseline'):
                e.lda_baseline()
            if Args.func.get('LDA_pairwise'):
                e.lda_pairwise()
            if Args.func.get('LDA_channel'):
                e.lda_channel()
            if Args.func.get('train'):
                e.train()
            if Args.func.get('analyse_kernels'):
                e.analyse_kernels()
            if Args.func.get('plot_kernels'):
                e.plot_kernels()
            if Args.func.get('generate'):
                e.generate()
            if Args.func.get('recursive'):
                e.recursive()
            if Args.func.get('kernel_network_FIR'):
                e.kernel_network_FIR()
            if Args.func.get('kernel_network_IIR'):
                e.kernel_network_IIR()
            if Args.func.get('feature_importance'):
                e.feature_importance()
            if Args.func.get('save_validation_subs'):
                e.save_validation_subs()
            if Args.func.get('save_validation_ch'):
                e.save_validation_channels()
            if Args.func.get('pca_sensor_loss'):
                e.pca_sensor_loss()
            if Args.func.get('compare_layers'):
                e.compare_layers()
            if Args.func.get('test'):
                e.test()
            if Args.func.get('LDA_eval'):
                e.lda_eval()
            if Args.func.get('lda_eval_train'):
                e.lda_eval_train()
            if Args.func.get('window_eval'):
                e.window_eval()
            if Args.func.get('PFIts'):
                e.PFIts()
            if Args.func.get('PFIch'):
                e.PFIch()
            if Args.func.get('PFIemb'):
                e.PFIemb()
            if Args.func.get('PFIfreq'):
                e.PFIfreq()
            if Args.func.get('PFIfreq_ch'):
                e.PFIfreq_ch()
            if Args.func.get('PFIfreq_ts'):
                e.PFIfreq_ts()
            if Args.func.get('gradient_analysis'):
                e.gradient_analysis()
            if Args.func.get('model_inversion'):
                e.model_inversion()
            if Args.func.get('multi2pair'):
                e.multi2pair()
            if Args.func.get('confusion_matrix'):
                e.confusion_matrix()
            if Args.func.get('save_validation_timepoints'):
                e.save_validation_timepoints()
            if Args.func.get('lda_eval_train_ensemble'):
                e.lda_eval_train_ensemble()
            if Args.func.get('LDA_crossval'):
                e.lda_crossval()
            if Args.func.get('lda_crossval_pairs'):
                e.lda_crossval_pairs()
            if Args.func.get('lda_eval_train_subs'):
                e.lda_eval_train_subs()
            if Args.func.get('evaluate_train'):
                e.evaluate_train()
            if Args.func.get('classify'):
                e.classify()

            e.save_embeddings()

            # delete model and dataset
            del e.model
            del e.dataset
            torch.cuda.empty_cache()
