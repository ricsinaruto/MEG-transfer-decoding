import os
import sys
import random
import numpy as np
from scipy.io import savemat, loadmat
import torch
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DondersData:
    '''
    Base class for loading and processing resting data and specifically from
    the Donders MOUS dataset.
    '''
    def __init__(self, args):
        '''
        Load data and apply pca, then create batches.
        '''
        self.inds = {'train': [], 'val': []}
        self.args = args

        sr = args.sample_rate[1] - args.sample_rate[0]
        self.shift = sr - args.timesteps - args.rf + 1
        self.mean = None
        self.var = None
        self.sub_id = {'train': [], 'val': []}

        # load normalization coefficients
        if not args.save_norm:
            norm = pickle.load(open(args.norm_path, 'rb'))
            self.mean = norm['means']
            self.var = norm['vars']

        # whether to load an already created PCA model
        if args.load_pca:
            pca_model = pickle.load(open(args.pca_path, 'rb'))
            self.pca_model = pca_model

        # load pickled data directly, no further processing required
        if args.load_data:
            self.load_mat_data(args)
            args.num_channels = len(args.num_channels)
            self.set_common()
            return

        # load the raw subject data
        x_trains, x_vals, disconts = self.load_data(args)

        # this is the continuous data for AR models
        self.x_train = np.concatenate(tuple(x_trains), axis=1)
        self.x_val = np.concatenate(tuple(x_vals), axis=1)

        # fit a new PCA model and save it to disk
        if args.num_components and not args.load_pca:
            pca_model = PCA(args.num_components, random_state=69)
            pca_model.fit(self.x_train.transpose())
            pickle.dump(pca_model, open(args.pca_path, 'wb'))
            args.num_channels = args.num_components

        # reduce number of channels with PCA model and normalize both splits
        if args.num_components or args.load_pca:
            print(pca_model.explained_variance_ratio_)
            print('Explained variance: ',
                  np.sum(pca_model.explained_variance_ratio_))

            self.x_train = pca_model.transform(
                self.x_train.transpose()).transpose()
            x_val = pca_model.transform(self.x_val.transpose()).transpose()
            args.num_channels = args.num_components

            # compute inverse transform to see reconstruction error
            x_rec = pca_model.inverse_transform(x_val.transpose())
            plt.plot(self.x_val[0, :2000])
            plt.plot(x_rec[:2000, 0])
            plt.savefig(os.path.join(args.result_dir, 'pca.svg'),
                        format='svg', dpi=1200)
            plt.close('all')
            self.x_val = x_val

            # normalize train and validation splits
            self.x_train, _, _ = self.normalize(
                self.x_train, self.mean, self.var)
            self.x_val, _, _ = self.normalize(self.x_val, self.mean, self.var)

            # save the means and variances of data
            if args.save_norm:
                norm = {'means': self.mean, 'vars': self.var}
                pickle.dump(norm, open(args.norm_path, 'wb'))

        # create examples from continuous data
        train_eps = []
        val_eps = []
        loop_iter = zip(x_trains, x_vals, disconts)
        for sid, (x_train, x_val, discont) in enumerate(loop_iter):
            if args.num_components or args.load_pca:
                # transform and normalize separately
                x_train = pca_model.transform(x_train.transpose()).transpose()
                x_val = pca_model.transform(x_val.transpose()).transpose()
                x_train, _, _ = self.normalize(x_train, self.mean, self.var)
                x_val, _, _ = self.normalize(x_val, self.mean, self.var)

            if discont[0] != 0:
                discont = [0] + discont

            # create examples by taking into account discontinuities
            val_ln = len(x_val[0])
            val_disconts = [i for i in discont if i < val_ln]
            examples = self.create_examples(x_val, val_disconts)
            val_eps.append(examples)
            self.sub_id['val'].extend([sid] * examples.shape[0])

            train_disconts = [0] + [i - val_ln for i in discont if i >= val_ln]
            examples = self.create_examples(x_train, train_disconts)
            train_eps.append(examples)
            self.sub_id['train'].extend([sid] * examples.shape[0])

        # concatenate across subjects and shuffle examples
        train_ep = np.concatenate(tuple(train_eps))
        val_ep = np.concatenate(tuple(val_eps))

        shuffled = list(range(train_ep.shape[0]))
        random.shuffle(shuffled)
        self.x_train_t = train_ep[shuffled, :, :]
        self.sub_id['train'] = np.array(self.sub_id['train'])[shuffled]

        shuffled = list(range(val_ep.shape[0]))
        random.shuffle(shuffled)
        self.x_val_t = val_ep[shuffled, :, :]
        self.sub_id['val'] = np.array(self.sub_id['val'])[shuffled]

        print('Good samples: ', sum([x.shape[1] for x in x_trains + x_vals]))
        print('Extracted samples: ',
              (train_ep.shape[0] + val_ep.shape[0]) * self.shift)

        if not os.path.isdir(os.path.split(args.dump_data)[0]):
            os.mkdir(os.path.split(args.dump_data)[0])

        self.save_data()
        self.set_common()

    def save_data(self):
        '''
        Save final data to disk for easier loading next time.
        '''
        for i in range(self.x_train.shape[0]):
            dump = {'x_train': self.x_train[i:i+1, :],
                    'x_val': self.x_val[i:i+1, :],
                    'x_train_t': self.x_train_t[:, i:i+1:, :],
                    'x_val_t': self.x_val_t[:, i:i+1, :],
                    'sub_id_train': self.sub_id['train'],
                    'sub_id_val': self.sub_id['val']}
            savemat(self.args.dump_data + 'ch' + str(i) + '.mat', dump)

    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        '''
        chn = args.num_channels
        data = loadmat(args.load_data)
        self.x_train = np.array(data['x_train'])[chn, :]
        self.x_val = np.array(data['x_val'])[chn, :]
        self.x_train_t = np.array(data['x_train_t'])[:, chn, :]
        self.x_val_t = np.array(data['x_val_t'])[:, chn, :]

    def get_batch(self, i, data, split='train'):
        '''
        Get batch with index i from dataset data.
        '''
        bs = self.bs[split]
        end = data.shape[0] if (i+1)*bs > data.shape[0] else (i+1)*bs
        return data[i*bs:end, :, :], self.sub_id[split][i*bs:end]

    def get_train_batch(self, i):
        # helper for getting a training batch
        return self.get_batch(i, self.x_train_t, 'train')

    def get_val_batch(self, i):
        # helper for getting a validation batch
        return self.get_batch(i, self.x_val_t, 'val')

    def find_bs(self, bs, shape):
        '''
        Convinience function for setting the batch size.
        '''
        if self.args.pred:
            return bs

        # if dealing with epoched data we ideally want to include all epochs
        for i in range(bs):
            if not shape % (bs - i):
                return bs - i

    def set_common(self):
        # set common parameters
        bs = self.args.batch_size
        self.bs = {'train': self.find_bs(bs, self.x_train_t.shape[0]),
                   'val': self.find_bs(bs, self.x_val_t.shape[0])}

        print('Train batch size: ', self.bs['train'])
        print('Validation batch size: ', self.bs['val'])

        self.train_batches = int(self.x_train_t.shape[0] / self.bs['train'])
        self.val_batches = int(self.x_val_t.shape[0] / self.bs['val'])

        try:
            self.x_train_t = torch.Tensor(self.x_train_t).float().cuda()
            self.x_val_t = torch.Tensor(self.x_val_t).float().cuda()
            self.sub_id['train'] = torch.LongTensor(self.sub_id['train']).cuda()
            self.sub_id['val'] = torch.LongTensor(self.sub_id['val']).cuda()
        except:
            self.x_train_t = torch.Tensor(self.x_train_t).float()
            self.x_val_t = torch.Tensor(self.x_val_t).float()
            self.sub_id['train'] = torch.LongTensor(self.sub_id['train'])
            self.sub_id['val'] = torch.LongTensor(self.sub_id['val'])

        self.sub_id['train'] = self.sub_id['train'].reshape(-1)
        self.sub_id['val'] = self.sub_id['val'].reshape(-1)

    def normalize(self, x, mean=None, var=None):
        '''
        Normalize x with optionally given mean and variance (var).
        '''
        x = x.transpose()
        mean = np.mean(x, axis=0) if mean is None else mean
        var = np.std(x, axis=0) if var is None else var
        x = (x - mean)/var
        return x.transpose(), mean, var

    def load_data(self, args):
        '''
        Load raw data from multiple subjects.
        '''
        # whether we are working with one subject or a directory of them
        if 'sub' in args.data_path:
            paths = [args.data_path]
        else:
            paths = os.listdir(args.data_path)
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if os.path.isdir(p)]

        x_trains = []
        x_vals = []
        disconts = []
        for path in paths:
            print(path)
            mask_path = os.path.join(path, 'good_samples_new.mat')
            mask = np.array(loadmat(mask_path)['X'])

            d = []
            # calculate discontinuity indices from the mask of good timesteps
            for i in range(len(mask)):
                if mask[i] == 1 and mask[i-1] == 0:
                    d.append(i - sum(abs(mask[:i] - 1)))
            disconts.append(d)

            data_path = os.path.join(path, 'preprocessed_data_new.mat')
            x_train = np.array(loadmat(data_path)['X'])[args.num_channels, :]
            x_train = x_train[:, mask.nonzero()[0]]

            # create training and validation splits
            x_val = x_train[:, :int(args.split * x_train.shape[1])]
            x_train = x_train[:, int(args.split * x_train.shape[1]):]

            x_train, mean, var = self.normalize(x_train)
            x_val, _, _ = self.normalize(x_val, mean, var)

            x_trains.append(x_train)
            x_vals.append(x_val)

        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, disconts

    def create_examples(self, x, disconts):
        '''
        Create examples from the continuous data (x) taking into account
        the discontinuities (disconts).
        '''
        # each element in x_segments is a continuous data segment
        x_segments = []
        if len(disconts) > 1:
            for i, m in enumerate(disconts[:-1]):
                if len(x[0, m:disconts[i + 1]]) > self.args.sample_rate:
                    x_segments.append(x[:, m:disconts[i + 1]])
            x_segments.append(x[:, disconts[-1]:])
        else:
            x_segments.append(x)

        # create samples with input size 'sample_rate', and shifting by 'shift'
        x_epochs = []
        for x in x_segments:
            i = 0
            samples = []
            while True:
                end = i * self.shift + self.args.sample_rate
                if end > x.shape[1]:
                    break
                samples.append(x[:, i * self.shift:end])
                i = i + 1

            x_epochs.extend(samples)

        x_epochs = np.array(x_epochs)
        #np.random.shuffle(x_epochs)

        return x_epochs
