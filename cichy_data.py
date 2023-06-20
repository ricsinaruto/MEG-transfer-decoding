import os
import torch
import random
import pickle
import mne
import dill
import math
import traceback
import numpy as np
import faiss
from sklearn.cluster import KMeans

from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA

from mrc_data import MRCData


def mulaw_inv(x, mu=255):
    '''
    Inverse mu-law companding.
    '''
    shape = x.shape

    x = x.reshape(-1)
    x = (x - 0.5) / mu * 2 - 1
    x = torch.sign(x)*((1+mu)**torch.abs(x)-1)/mu

    x = x.reshape(shape)
    return x


class CichyData(MRCData):
    '''
    Class for loading the trials from the Cichy dataset.
    '''
    def set_subjects(self, split):
        inds = np.in1d(self.sub_id[split], self.args.subjects_data)
        self.sub_id[split] = self.sub_id[split][:, inds]

        if split == 'train':
            self.x_train_t = self.x_train_t[inds]
        elif split == 'val':
            self.x_val_t = self.x_val_t[inds]
        elif split == 'test':
            self.x_test_t = self.x_test_t[inds]

    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        '''
        chn = args.num_channels
        x_train_ts = []
        x_val_ts = []
        x_test_ts = []

        # load data for each channel
        for index, i in enumerate(chn):
            data = loadmat(args.load_data + 'ch' + str(i) + '.mat')
            x_train_ts.append(np.array(data['x_train_t']))
            x_val_ts.append(np.array(data['x_val_t']))
            try:
                x_test_ts.append(np.array(data['x_test_t']))
            except:
                pass

            if index == 0:
                self.sub_id['train'] = np.array(data['sub_id_train'])
                self.sub_id['val'] = np.array(data['sub_id_val'])
                try:
                    self.sub_id['test'] = np.array(data['sub_id_test'])
                except:
                    pass

        self.x_train_t = np.concatenate(tuple(x_train_ts), axis=1)
        self.x_val_t = np.concatenate(tuple(x_val_ts), axis=1)

        if len(x_test_ts) == 0:
            self.x_test_t = self.x_val_t
            self.sub_id['test'] = self.sub_id['val']
        else:
            self.x_test_t = np.concatenate(tuple(x_test_ts), axis=1)

    def trial_subset(self, data, args):
        '''
        Selects a subset of trials from the data.
        '''
        num_ch = len(args.num_channels) - 1
        # select a subset of training trials
        num_trials = np.sum(data[:, num_ch, 0] == 0.0)
        max_trials = int(args.max_trials * num_trials)
        trials = [0] * args.num_classes

        inds = []
        for i in range(data.shape[0]):
            cond = int(data[i, num_ch, 0])
            if trials[cond] < max_trials:
                trials[cond] += 1
                inds.append(i)

        return inds

    def set_common(self, args):
        if not isinstance(args.num_channels, list):
            args.num_channels = list(range(args.num_channels+1))

        num_ch = len(args.num_channels) - 1

        # select wanted subjects
        if args.subjects_data:
            self.set_subjects('train')
            self.set_subjects('val')
            self.set_subjects('test')

        # crop data
        tmin = args.sample_rate[0]
        tmax = args.sample_rate[1]
        self.x_train_t = self.x_train_t[:, :, tmin:tmax]
        self.x_val_t = self.x_val_t[:, :, tmin:tmax]
        self.x_test_t = self.x_test_t[:, :, tmin:tmax]

        args.sample_rate = tmax - tmin

        inds = self.trial_subset(self.x_train_t, args)
        self.x_train_t = self.x_train_t[inds, :, :]

        if args.val_max_trials:
            inds = self.trial_subset(self.x_val_t, args)
            self.x_val_t = self.x_val_t[inds, :, :]
            self.x_test_t = self.x_test_t[inds, :, :]

        # whiten data if needed
        if args.group_whiten:
            # reshape for PCA
            x_train = self.x_train_t[:, :num_ch, :].transpose(0, 2, 1)
            x_val = self.x_val_t[:, :num_ch, :].transpose(0, 2, 1)
            x_test = self.x_test_t[:, :num_ch, :].transpose(0, 2, 1)
            x_train = x_train.reshape(-1, num_ch)
            x_val = x_val.reshape(-1, num_ch)
            x_test = x_test.reshape(-1, num_ch)

            # change dim red temporarily
            dim_red = args.dim_red
            args.dim_red = num_ch
            x_train, x_val, x_test = self.whiten(x_train, x_val, x_test)
            args.dim_red = dim_red

            # reshape back to trials
            x_train = x_train.reshape(-1, args.sample_rate, num_ch)
            x_val = x_val.reshape(-1, args.sample_rate, num_ch)
            x_test = x_test.reshape(-1, args.sample_rate, num_ch)
            x_train = x_train.transpose(0, 2, 1)
            x_val = x_val.transpose(0, 2, 1)
            x_test = x_test.transpose(0, 2, 1)

            self.x_train_t[:, :num_ch, :] = x_train
            self.x_val_t[:, :num_ch, :] = x_val
            self.x_test_t[:, :num_ch, :] = x_test

        args.num_channels = args.num_channels[:-1]

        super(CichyData, self).set_common()

    def save_data(self):
        '''
        Save final data to disk for easier loading next time.
        '''
        if self.args.save_data:
            for i in range(self.x_train_t.shape[1]):
                dump = {'x_train_t': self.x_train_t[:, i:i+1:, :],
                        'x_val_t': self.x_val_t[:, i:i+1, :],
                        'x_test_t': self.x_test_t[:, i:i+1, :],
                        'sub_id_train': self.sub_id['train'],
                        'sub_id_val': self.sub_id['val'],
                        'sub_id_test': self.sub_id['test']}

                savemat(self.args.dump_data + 'ch' + str(i) + '.mat', dump)

        # save standardscaler
        path = os.path.join('/'.join(self.args.dump_data.split('/')[:-1]),
                            'standardscaler')
        with open(path, 'wb') as file:
            pickle.dump(self.norm, file)

    def splitting(self, dataset, args):
        split_l = int(args.split[0] * dataset.shape[1])
        split_h = int(args.split[1] * dataset.shape[1])
        x_val = dataset[:, split_l:split_h, :, :]
        x_train = dataset[:, :split_l, :, :]
        x_train = np.concatenate((x_train, dataset[:, split_h:, :, :]),
                                 axis=1)

        return x_train, x_val, x_val

    def load_data(self, args):
        '''
        Load trials for each condition from multiple subjects.
        '''
        # whether we are working with one subject or a directory of them
        if isinstance(args.data_path, list):
            paths = args.data_path
        elif 'sub' in args.data_path:
            paths = [args.data_path]
        else:
            paths = os.listdir(args.data_path)
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if os.path.isdir(p)]
            paths = [p for p in paths if 'opt' not in p]
            paths = [p for p in paths if 'sub' in p]

        channels = len(args.num_channels)
        x_trains = []
        x_vals = []
        x_tests = []
        for path in paths:
            print('Loading ', path, flush=True)
            min_trials = 1000000
            dataset = []

            # loop over conditions
            for c in range(args.num_classes):
                cond_path = os.path.join(path, 'cond' + str(c))
                files = os.listdir(cond_path)
                files = [f for f in files if 'npy' in f]
                if len(files) < min_trials:
                    min_trials = len(files)

            for c in range(args.num_classes):
                cond_path = os.path.join(path, 'cond' + str(c))
                trials = []
                # loop over trials within a condition
                for i in range(min_trials):
                    trial = np.load(os.path.join(cond_path, f'trial{i}.npy'))
                    trials.append(trial)

                dataset.append(np.array(trials))

            # condition with lowest number of trials
            print('Minimum trials: ', min_trials, flush=True)

            # dataset shape: conditions x trials x timesteps x channels
            dataset = np.array([t[:min_trials, :, :] for t in dataset])

            if args.whiten > 1000:
                args.num_channels = list(range(dataset.shape[-1]))
                args.whiten = dataset.shape[-1]
                channels = dataset.shape[-1]

            # choose first 306 channels
            dataset = dataset.transpose(0, 1, 3, 2)
            if hasattr(args, 'flip_axes'):
                if args.flip_axes:
                    dataset = dataset.transpose(0, 1, 3, 2)

            dataset = dataset[:, :, args.num_channels, :]
            self.timesteps = dataset.shape[3]

            # create training and validation splits with equal class numbers
            x_train, x_val, x_test = self.splitting(dataset, args)

            # crop training trials
            max_trials = round(args.max_trials * x_train.shape[1])
            x_train = x_train[:, :max_trials, :, :]

            print(x_train.shape)
            x_train = x_train.transpose(0, 1, 3, 2).reshape(-1, channels)
            x_val = x_val.transpose(0, 1, 3, 2).reshape(-1, channels)
            x_test = x_test.transpose(0, 1, 3, 2).reshape(-1, channels)

            # standardize dataset along channels
            x_train, x_val, x_test = self.normalize(x_train, x_val, x_test)

            x_trains.append(x_train)
            x_vals.append(x_val)
            x_tests.append(x_test)

        # this is just needed to work together with other dataset classes
        disconts = [[0] for path in paths]
        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, x_tests, disconts

    def normalize_ex(self, data):
        '''
        data = data.transpose(0, 1, 3, 2)
        scaler = StandardScaler()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j, :, :] = scaler.fit_transform(data[i, j, :, :])

        return data.transpose(0, 1, 3, 2)
        '''

        trials = data.shape[1]
        data = data.reshape(-1, data.shape[2], data.shape[3])
        data = data.astype(np.float64)
        data = mne.filter.notch_filter(
            data, 1000, np.array([50, 100, 150]), phase='minimum')
        data = mne.filter.filter_data(
            data, 1000, 0.1, 124.9, phase='minimum')

        return data.reshape(-1, trials, data.shape[1], data.shape[2])

    def create_examples(self, x, disconts):
        '''
        Create examples with labels.
        '''

        # expand shape to trials
        x = x.transpose(1, 0)
        x = x.reshape(self.args.num_classes, -1, self.timesteps, x.shape[1])
        x = x.transpose(0, 1, 3, 2)

        # downsample data if needed
        resample = int(self.args.original_sr/self.args.sr_data)
        x = x[:, :, :, ::resample]
        timesteps = x.shape[3]
        trials = x.shape[1]

        # create labels, and put them in the last channel of the data
        array = []
        labels = np.ones((trials, 1, timesteps))
        for c in range(x.shape[0]):
            array.append(np.concatenate((x[c, :, :, :], labels * c), axis=1))

        x = np.array(array).reshape(-1, x.shape[2] + 1, timesteps)
        return x


class CichyDataDISP(CichyData):
    def splitting(self, dataset, args):
        split_l = int(args.split[0] * dataset.shape[1])
        split_h = int(args.split[1] * dataset.shape[1])
        x_val = dataset[:, split_l:split_h, :, :]

        x_train_lower, x_train_upper = None, None
        if split_l > 1:
            x_train_lower = dataset[:, :split_l-1, :, :]
        if split_h < dataset.shape[1] - 1:
            x_train_upper = dataset[:, split_h+1:, :, :]

        if x_train_lower is not None and x_train_upper is not None:
            x_train = np.concatenate((x_train_lower, x_train_upper),
                                     axis=1)
        elif x_train_lower is not None:
            x_train = x_train_lower
        elif x_train_upper is not None:
            x_train = x_train_upper

        return x_train, x_val, x_val


class CichyDataTrialNorm(CichyData):
    def normalize(self, x_train, x_val, x_test):
        '''
        Standardize and whiten data if needed.
        '''
        # standardize dataset along channels
        self.norm = StandardScaler()

        resample = int(self.args.original_sr/self.args.sr_data)

        # expand shape to trials x timesteps x channels
        x_train = x_train.reshape(-1, self.timesteps, x_train.shape[1])
        x_val = x_val.reshape(-1, self.timesteps, x_val.shape[1])
        x_test = x_test.reshape(-1, self.timesteps, x_test.shape[1])

        x_train = x_train[:, ::resample, :]
        x_val = x_val[:, ::resample, :]
        x_test = x_test[:, ::resample, :]
        self.timesteps = x_train.shape[1]

        # squeeze to trials x (timesteps x channels)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        self.norm.fit(x_train)
        print(x_train.shape)

        #x_train = self.norm.transform(x_train)
        #x_val = self.norm.transform(x_val)
        #x_test = self.norm.transform(x_test)
        var = 1e-6
        var = np.std(x_train)
        print('Global variance: ', var)

        mean = np.mean(x_train, axis=0)
        x_train = (x_train - mean)/var
        x_val = (x_val - mean)/var
        x_test = (x_test - mean)/var

        # expand shape to trials x timesteps x channels
        x_train = x_train.reshape(x_train.shape[0], self.timesteps, -1)
        x_val = x_val.reshape(x_val.shape[0], self.timesteps, -1)
        x_test = x_test.reshape(x_test.shape[0], self.timesteps, -1)

        # squeeze to (trials x timesteps) x channels
        x_train = x_train.reshape(-1, x_train.shape[2])
        x_val = x_val.reshape(-1, x_val.shape[2])
        x_test = x_test.reshape(-1, x_test.shape[2])

        return x_train.T, x_val.T, x_test.T

    def create_examples(self, x, disconts):
        '''
        Create examples with labels.
        '''

        # expand shape to trials
        x = x.transpose(1, 0)
        x = x.reshape(self.args.num_classes, -1, self.timesteps, x.shape[1])
        x = x.transpose(0, 1, 3, 2)

        # downsample data if needed
        timesteps = x.shape[3]
        trials = x.shape[1]

        # create labels, and put them in the last channel of the data
        array = []
        labels = np.ones((trials, 1, timesteps))
        for c in range(x.shape[0]):
            array.append(np.concatenate((x[c, :, :, :], labels * c), axis=1))

        x = np.array(array).reshape(-1, x.shape[2] + 1, timesteps)
        return x


class CichyDataDISPTrialNorm(CichyDataDISP, CichyDataTrialNorm):
    pass


class CichyDataCHN(CichyData):
    '''
    Same as CichyData, but channel numbers are not pre-specified.
    '''
    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        Number of channels is inferred from number of files.
        '''

        # get number of channels by counting files in args.load_data
        chn = []
        for f in os.listdir(args.data_path):
            if 'ch' in f:
                chn.append(int(f.split('.')[0].split('ch')[-1]))

        args.num_channels = sorted(chn)
        super().load_mat_data(args)


class CichyDataRandsample(CichyData):
    '''
    Class for randomizing the batch retrieval for the Cichy dataset.
    '''
    def get_train_batch(self, i):
        if i == 0:
            self.inds['train'] = list(range(self.x_train_t.shape[0]))

        # sample random indices
        inds = random.sample(self.inds['train'], self.bs['train'])
        self.inds['train'] = [v for v in self.inds['train'] if v not in inds]

        return self.x_train_t[inds, :, :], self.sub_id['train'][inds]


class CichyDataRobust(CichyData):
    def product_quantize(self, x_train, x_val, x_test):

        # convert to contiguous array
        x_train = np.ascontiguousarray(x_train, dtype=np.float32)
        x_val = np.ascontiguousarray(x_val, dtype=np.float32)
        x_test = np.ascontiguousarray(x_test, dtype=np.float32)
        '''
        self.res_quantizer = faiss.ResidualQuantizer(x_train.shape[1], 2, 8)
        self.res_quantizer.train(x_train)

        # encode
        codes = self.res_quantizer.compute_codes(x_train)
        x_train = self.res_quantizer.decode(codes)

        codes = self.res_quantizer.compute_codes(x_val)
        x_val = self.res_quantizer.decode(codes)

        codes = self.res_quantizer.compute_codes(x_test)
        x_test = self.res_quantizer.decode(codes)
        '''

        # put the 306 channels into 6 buckets based on covariance
        # in each bucket the channels should have similar covariances

        num_buckets = 40

        # Compute the covariance matrix of the features
        cov_matrix = np.cov(x_train, rowvar=False)

        # Apply K-means clustering on the covariance matrix
        kmeans = KMeans(n_clusters=num_buckets, random_state=0).fit(cov_matrix)

        # Create a dictionary to store the features in each bucket
        buckets = {i: [] for i in range(num_buckets)}

        # Assign features to the corresponding buckets
        for feature_idx, bucket in enumerate(kmeans.labels_):
            buckets[bucket].append(feature_idx)


        num_subspaces = 2  # Number of subspaces
        num_clusters_per_subspace = 6  # Total quantization bins = num_clusters_per_subspace ** num_subspaces = 100k

        # create a residual quantizer for each bucket of features
        for bucket, features in buckets.items():
            res_quant = faiss.ResidualQuantizer(len(features), num_subspaces, num_clusters_per_subspace)
            res_quant.train(np.ascontiguousarray(x_train[:, features]))

            xt = np.ascontiguousarray(x_train[:, features], dtype=np.float32)
            codes = res_quant.compute_codes(xt)
            x_train[:, features] = res_quant.decode(codes)

            xv = np.ascontiguousarray(x_val[:, features], dtype=np.float32)
            codes = res_quant.compute_codes(xv)
            x_val[:, features] = res_quant.decode(codes)

            xt = np.ascontiguousarray(x_test[:, features], dtype=np.float32)
            codes = res_quant.compute_codes(xt)
            x_test[:, features] = res_quant.decode(codes)

        self.norm2 = RobustScaler()
        self.norm2.fit(x_train)
        x_train = self.norm2.transform(x_train)
        x_val = self.norm2.transform(x_val)
        x_test = self.norm2.transform(x_test)

        return x_train, x_val, x_test
    
    def save_data(self):
        super().save_data()

        '''
        # check if self.res_quantizer and self.norm2 exist
        if hasattr(self, 'res_quantizer') and hasattr(self, 'norm2'):
            # save res_quantizer and norm2
            path = os.path.join('/'.join(self.args.dump_data.split('/')[:-1]),
                                'quantizer')
            with open(path, 'wb') as file:
                pickle.dump(self.res_quantizer, file)

            path = os.path.join('/'.join(self.args.dump_data.split('/')[:-1]),
                                'norm2')
            with open(path, 'wb') as file:
                pickle.dump(self.norm2, file)
        '''

    def normalize(self, x_train, x_val, x_test):
        '''
        Standardize and whiten data if needed.
        '''
        # standardize dataset along channels
        self.norm = RobustScaler()
        self.norm.fit(x_train)
        print(x_train.shape)
        x_train = self.norm.transform(x_train)
        x_val = self.norm.transform(x_val)
        x_test = self.norm.transform(x_test)

        # if needed, remove covariance with PCA
        if self.args.whiten:
            x_train, x_val, x_test = self.whiten(x_train, x_val, x_test)

        # check if args has product_quant attribute
        if hasattr(self.args, 'product_quant'):
            if self.args.product_quant:
                x_train, x_val, x_test = self.product_quantize(
                    x_train, x_val, x_test)

        return x_train.T, x_val.T, x_test.T


class CichyDataCrossval(CichyDataRobust):
    def splitting(self, dataset, args):
        split = args.split[1] - args.split[0]
        split = int(split*dataset.shape[1])

        if args.shuffle:
            for i in range(dataset.shape[0]):
                perm = np.random.permutation(dataset.shape[1])
                dataset[i, :, :, :] = dataset[i, perm, :, :]

        # create separate val and test splits
        x_val = dataset[:, :split, :, :]
        x_train = dataset[:, split:, :, :]
        x_test = x_train[:, :split:, :, :]
        x_train = x_train[:, split:, :, :]

        return x_train, x_val, x_test


class CichyDataNoNorm(CichyData):
    def normalize(self, x_train, x_val, x_test):
        self.norm = RobustScaler()
        return x_train.T, x_val.T, x_test.T


class CichyDataCrossvalRobust(CichyDataCrossval, CichyDataRobust):
    pass


class CichyDataCrossvalNoNorm(CichyDataCrossval):
    def normalize(self, x_train, x_val, x_test):
        self.norm = StandardScaler()
        return x_train.T, x_val.T, x_test.T


class CichyContData(MRCData):
    '''
    Implements the continuous classification problem on the Cichy dataset.
    Under construction.
    '''
    def load_data(self, args):
        '''
        Load raw data from multiple subjects.
        '''
        # whether we are working with one subject or a directory of them
        if isinstance(args.data_path, list):
            paths = args.data_path
        else:
            paths = os.listdir(args.data_path)
            paths = [p for p in paths if 'sub' in p]
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if not os.path.isdir(p)]
        print('Number of subjects: ', len(paths))

        resample = int(1000/args.sr_data)
        epoch_len = int(0.5*args.sr_data)

        # split ratio
        split = args.split[1] - args.split[0]
        split_trials = int(30 * split)

        x_trains = []
        x_vals = []
        disconts = []
        for path in paths:
            print(path)

            # will have to be changed to handle concatenated subjects
            # load event timings for continuous data
            ev_path = os.path.join(args.data_path, 'event_times.npy')
            event_times = np.load(ev_path)
            event_times = [(int(ev[0]/resample), ev[2]) for ev in event_times]

            dataset = np.load(path).T

            # choose first 306 channels and downsample
            dataset = dataset[args.num_channels, ::resample]
            labels = [118] * dataset.shape[1]

            val_counter = [0] * args.num_classes
            val_events = []
            test_events = []
            train_events = []
            '''
                    elif val_counter[ev[1]-1] < split_trials * 2:
                        val_counter[ev[1]-1] += 1
                        test_events.append(ev[0])
            '''
            # set labels
            for ev in event_times:
                if ev[1] < 119:
                    labels[ev[0]:ev[0]+epoch_len] = [ev[1]-1] * epoch_len

                    if val_counter[ev[1]-1] < 6:
                        val_counter[ev[1]-1] += 1
                        val_events.append(ev[0])
                    else:
                        train_events.append(ev[0])

            labels = np.array(labels)

            print('Last val sample: ', max(val_events))
            print('First train sample: ', min(train_events))

            split = int((max(val_events) + min(train_events))/2)
            labels = {'val': labels[:split].reshape(1, -1),
                      'train': labels[split:].reshape(1, -1)}

            # create training and validation splits
            x_val = dataset[:, :split]
            x_test = dataset[:, :split]
            x_train = dataset[:, split:]

            x_train, x_val, x_test = self.normalize(x_train.T,
                                                    x_val.T,
                                                    x_test.T)

            # add labels to data
            x_val = np.concatenate((x_val, labels['val']), axis=0)
            x_train = np.concatenate((x_train, labels['train']), axis=0)

            x_trains.append(x_train)
            x_vals.append(x_val)

        # this is just needed to work together with other dataset classes
        disconts = [[0] for path in paths]
        return x_trains, x_vals, x_vals, disconts

    def create_examples(self, x, disconts):
        '''
        Create examples with labels.
        '''
        return x.reshape(1, x.shape[0], x.shape[1])

    def set_common(self, args=None):
        # set common parameters
        super(CichyContData, self).set_common()

        sr = self.args.sample_rate
        bs = self.args.batch_size
        self.bs = {'train': bs, 'val': bs, 'test': bs}

        print('Train batch size: ', self.bs['train'])
        print('Validation batch size: ', self.bs['val'])

        self.train_batches = int(
            (self.x_train_t.shape[2] - sr - 1) / self.bs['train'])
        self.val_batches = int(
            (self.x_val_t.shape[2] - sr - 1) / self.bs['val'])
        self.test_batches = int(
            (self.x_test_t.shape[2] - sr - 1) / self.bs['test'])

        args.num_channels -= 1

    def get_batch(self, i, data, split):
        sr = self.args.sample_rate
        if i == 0:
            self.inds[split] = np.arange(data.shape[2] - sr)
            np.random.shuffle(self.inds[split])

        # sample random indices
        inds = self.inds[split][:self.bs[split]]
        data = torch.stack([data[0, :, ind:ind+sr] for ind in inds])

        self.inds[split] = self.inds[split][self.bs[split]:]
        return data, self.sub_id[split]

    def create_labels(self, data):
        sr = self.args.sample_rate[1] - self.args.sample_rate[0]
        inds = list(range(data.shape[2] - sr))
        peak_times = [-1] * data.shape[2]

        # how many timesteps needed to detect image
        im_len = int(0.5*self.args.sr_data)
        tresh = int(self.args.decode_front*self.args.sr_data)
        halftresh = int(self.args.decode_front*self.args.sr_data/2)
        back_tresh = int(self.args.decode_back*self.args.sr_data)
        peak = int(self.args.decode_peak*self.args.sr_data)

        im_presence = data[0, -1, :].copy()
        im_presence = (im_presence < 118).astype(int)

        # loop over examples in batch
        for i in inds:
            targets = data[0, -1, i:i+sr].copy()

            # if last timestep is one of the 118 images and enough time
            # has elapsed since image presentation
            if targets[-1] < 118 and np.all(targets[-tresh:] == targets[-1]):
                data[0, -1, i] = targets[-1]

                # set image peak time (150ms)
                tinds = np.nonzero(targets - targets[-1])[0]
                peak_times[i] = tinds[-1] + peak
                if tinds[-1] < sr - im_len - 1 or peak_times[i] > sr-1:
                    print('Error1')
                    print(targets)

            # else predict last image presented
            elif (targets[0] < 118 and
                    np.all(targets[:back_tresh] == targets[0])):
                data[0, -1, i] = targets[0]

                # set image peak time (150ms)
                tinds = np.nonzero(targets - targets[0])[0]
                peak_times[i] = tinds[0] - im_len + peak + 1
                if tinds[0] > im_len + 1 or peak_times[i] < 0:
                    print('Error2')
                    print(peak_times[i])
                    print(targets)

            # else predict image in middle of window
            elif targets[im_len-1] < 118 or targets[-im_len+1] < 118:
                data[0, -1, i] = targets[-im_len+1]
                if targets[im_len-1] < 118:
                    data[0, -1, i] = targets[im_len-1]

                # set image start
                tinds = np.nonzero(targets - 118)[0]
                tinds2 = np.nonzero(targets - targets[0])[0]
                image_start = tinds[tinds > tinds2[0]][0]
                peak_times[i] = image_start + peak
                if image_start < 0 or image_start + im_len > sr:
                    print('Error3')
                    print(targets)
            else:
                data[0, -1, i] = 118

            if peak_times[i] < -1 or peak_times[i] > sr - 1:
                print(targets)

            #print(data[0, -1, i])
            #print(targets)

        # scale non-epoch class
        nonepoch_trials = sum(data[0, -1, :] == 118)
        epoch_trials = sum(data[0, -1, :] == 0)
        self.args.epoch_ratio = epoch_trials / nonepoch_trials

        print('Non-epoch trials: ', sum(data[0, -1, :] == 118))
        print('Class 0  trials: ', sum(data[0, -1, :] == 0))
        print('Class 1  trials: ', sum(data[0, -1, :] == 1))
        print('Total trials: ', data.shape[2])

        peak_times = np.array(peak_times).reshape(1, 1, -1)
        im_presence = np.array(im_presence).reshape(1, 1, -1)
        return peak_times, im_presence

    def load_mat_data(self, args):
        super(CichyContData, self).load_mat_data(args)

        ev_times, im_presence = self.create_labels(self.x_train_t)
        self.x_train_t = np.concatenate((self.x_train_t[:, :-1, :],
                                         im_presence,
                                         ev_times,
                                         self.x_train_t[:, -1:, :]),
                                        axis=1)

        ev_times, im_presence = self.create_labels(self.x_val_t)
        self.x_val_t = np.concatenate((self.x_val_t[:, :-1, :],
                                       im_presence,
                                       ev_times,
                                       self.x_val_t[:, -1:, :]),
                                      axis=1)

        self.x_test_t = self.x_val_t


class CichySimpleContData(CichyContData):
    def create_labels(self, data):
        sr = self.args.sample_rate[1] - self.args.sample_rate[0]
        inds = list(range(data.shape[2] - sr))
        peak_times = [-1] * data.shape[2]

        im_presence = data[0, -1, :].copy()
        im_presence = (im_presence < 118).astype(int)
        im_presence = np.array(im_presence).reshape(1, 1, -1)

        # how many timesteps needed to detect image
        peak = int(self.args.decode_peak*self.args.sr_data)

        # loop over examples in batch
        for i in inds:
            targets = data[0, -1, i:i+sr].copy()

            end_loop = False
            starts = list(range(20, 40))
            for ind in starts:
                if targets[ind] < 118 and np.all(targets[ind:ind+125] == targets[ind]):
                    data[0, -1, i] = targets[ind]
                    peak_times[i] = ind + peak
                    end_loop = True

                if end_loop:
                    break

            if not end_loop:
                data[0, -1, i] = 118

        # scale non-epoch class
        nonepoch_trials = sum(data[0, -1, :] == 118)
        epoch_trials = sum(data[0, -1, :] == 0)
        self.args.epoch_ratio = epoch_trials / nonepoch_trials

        print('Non-epoch trials: ', sum(data[0, -1, :] == 118))
        print('Class 0  trials: ', sum(data[0, -1, :] == 0))
        print('Class 1  trials: ', sum(data[0, -1, :] == 1))
        print('Total trials: ', data.shape[2])

        peak_times = np.array(peak_times).reshape(1, 1, -1)
        return peak_times, im_presence


class CichyQuantized(MRCData):
    def __init__(self, args):
        '''
        Load data and apply pca, then create batches.
        '''
        self.args = args
        self.inds = {'train': [], 'val': [], 'test': []}
        self.sub_id = {'train': [0], 'val': [0], 'test': [0]}
        self.chn_weights_sample = {}
        self.chn_ids = {}

        # load pickled data directly, no further processing required
        if args.load_data:
            self.load_mat_data(args)
            self.set_common(args)
            return

        # load the raw subject data
        x_trains, x_vals, x_tests = self.load_data(args)

        # this is the continuous data for AR models
        self.x_train = np.concatenate(tuple(x_trains), axis=1)
        self.x_val = np.concatenate(tuple(x_vals), axis=1)
        self.x_test = np.concatenate(tuple(x_tests), axis=1)

        if not args.bypass:
            self.encode()

            os.makedirs(args.dump_data, exist_ok=True)

            self.save_data()
            self.set_common(args)
        else:
            args.num_channels = len(args.num_channels)

    def load_data(self, args):
        '''
        Load raw data from multiple subjects.
        '''
        # whether we are working with one subject or a directory of them
        if isinstance(args.data_path, list):
            paths = args.data_path
        else:
            paths = os.listdir(args.data_path)
            paths = [p for p in paths if 'sub' in p]
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if not os.path.isdir(p)]
        print('Number of subjects: ', len(paths))

        resample = int(1000/args.sr_data)
        epoch_len = int(0.5*args.sr_data)

        x_trains = []
        x_vals = []
        x_tests = []
        for sid, path in enumerate(paths):
            print(path)

            # will have to be changed to handle concatenated subjects
            # load event timings for continuous data
            ev_path = os.path.join(os.path.dirname(path), 'event_times.npy')
            event_times = np.load(ev_path)
            event_times = [(int(ev[0]/resample), ev[2]) for ev in event_times]

            dataset = np.load(path)

            # filter if needed
            if args.filter:
                iir_params = dict(order=5, ftype='butter')
                dataset = mne.filter.filter_data(dataset,
                                                 1000,
                                                 args.filter[0],
                                                 args.filter[1],
                                                 method='iir',
                                                 iir_params=iir_params)

            # choose first 306 channels and downsample
            dataset = dataset[args.num_channels, ::resample]
            labels = [0] * dataset.shape[1]

            val_counter = [0] * args.num_classes
            test_counter = [0] * args.num_classes
            val_events = []
            test_events = []
            train_events = []

            # set labels
            for ev in event_times:
                if ev[1] < 119:
                    cond = ev[1]-1
                    labels[ev[0]:ev[0]+epoch_len] = [cond+1] * epoch_len

                    if val_counter[cond] < 4:
                        val_counter[cond] += 1
                        val_events.append(ev[0])
                    elif test_counter[cond] < 4:
                        test_counter[cond] += 1
                        test_events.append(ev[0])
                    else:
                        train_events.append(ev[0])

            labels = np.array(labels)

            print('Last val sample: ', max(val_events))
            print('First test sample: ', min(test_events))
            print('Last test sample: ', max(test_events))
            print('First train sample: ', min(train_events))

            split_v = int((max(val_events) + min(test_events))/2)
            split_t = int((max(test_events) + min(train_events))/2)
            labels = {'val': labels[:split_v].reshape(1, -1),
                      'test': labels[split_v:split_t].reshape(1, -1),
                      'train': labels[split_t:].reshape(1, -1)}

            # create training and validation splits
            x_val = dataset[:, :split_v]
            x_test = dataset[:, split_v:split_t]
            x_train = dataset[:, split_t:]

            x_train, x_val, x_test = self.normalize(x_train, x_val, x_test)

            # add labels to data
            subid = np.array([sid] * x_val.shape[1]).reshape(1, -1)
            x_val = np.concatenate((x_val, labels['val'], subid), axis=0)

            subid = np.array([sid] * x_test.shape[1]).reshape(1, -1)
            x_test = np.concatenate((x_test, labels['test'], subid), axis=0)

            subid = np.array([sid] * x_train.shape[1]).reshape(1, -1)
            x_train = np.concatenate((x_train, labels['train'], subid), axis=0)

            x_trains.append(x_train)
            x_tests.append(x_test)
            x_vals.append(x_val)

        return x_trains, x_vals, x_tests

    def clip_(self, x):
        sorted_ = np.sort(x)
        clip_vals = sorted_[:, -self.args.num_clip]

        for i in range(x.shape[0]):
            x[i, :] = np.clip(x[i, :], -clip_vals[i], clip_vals[i])

        return x

    def normalize(self, xtn, xv, xtt):
        scaler = RobustScaler()
        xtn = scaler.fit_transform(xtn.T).T
        xv = scaler.transform(xv.T).T
        xtt = scaler.transform(xtt.T).T

        return xtn, xv, xtt

    def mulaw(self, x):
        '''
        Apply mu-law companding to input data.
        '''
        mu = self.args.mu
        shape = x.shape

        x = x.reshape(-1)
        x = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)

        digitized = ((x + 1) / 2 * mu+0.5).astype(np.int32)
        x = 2 * ((digitized).astype(np.float32) / mu) - 1

        x = x.reshape(shape)
        digitized = digitized.reshape(shape)

        x = np.append(digitized[:, :-1], digitized[:, 1:], axis=0)
        return x

    def encode(self):
        '''
        Encode data using mu-law companding.
        '''
        xtn = self.x_train[:-2]
        xv = self.x_val[:-2]
        xtt = self.x_test[:-2]

        xtn = np.clip(xtn, -self.args.num_clip, self.args.num_clip)

        self.maxabs = MaxAbsScaler()

        xtn = self.maxabs.fit_transform(xtn.T).T
        xv = self.maxabs.transform(xv.T).T
        xtt = self.maxabs.transform(xtt.T).T

        xv = np.clip(xv, -1, 1)
        xtt = np.clip(xtt, -1, 1)

        xtn = self.mulaw(xtn)
        xv = self.mulaw(xv)
        xtt = self.mulaw(xtt)

        self.x_train_t = np.append(xtn, self.x_train[-2:, :-1], axis=0)
        self.x_val_t = np.append(xv, self.x_val[-2:, :-1], axis=0)
        self.x_test_t = np.append(xtt, self.x_test[-2:, :-1], axis=0)

    def decode(self, x):
        '''
        Decode data by applying the inverse of mulaw and encoding functions.
        '''
        device = x.device

        x = mulaw_inv(x)

        # reshape x from (B, C, T) to (B*T, C)
        x = x.permute(0, 2, 1).contiguous().view(-1, x.shape[1])
        x = x.cpu().detach().numpy()

        x = self.maxabs.inverse_transform(x)

        x = torch.from_numpy(x).to(device)

        return x

    def _get_batch(self, i, data, split):
        '''
        Get batch of data.
        '''
        num_chn = self.args.num_channels
        model_chn_dim = num_chn
        if hasattr(self.args, 'model_chn_dim'):
            model_chn_dim = self.args.model_chn_dim

        if i == 0:
            self.inds[split] = np.random.permutation(data.shape[0])

        bs = min(self.bs[split], len(self.inds[split]))
        if bs == 0:
            # print(split, ' ', i)
            return None, None, None

        # sample random indices
        inds = self.inds[split][:bs]

        # place on correct gpu if not already there
        data = data[inds] if data.is_cuda else data[inds].to(self.args.device)

        # remove the already sampled indices
        self.inds[split] = self.inds[split][bs:]

        # loop over channel batching
        batch = []
        for j in range(int(num_chn/model_chn_dim)):
            targets = data[:, j*model_chn_dim+num_chn:
                           (j+1)*model_chn_dim+num_chn, :]

            # data: 306 input chs, 306 target chns, 1 condition id, 1 subject id
            d = {'inputs': data[:, j*model_chn_dim:(j+1)*model_chn_dim, :],
                 'targets': targets,
                 'condition': data[:, -2:-1, :],
                 'sid': data[:, -1:, :],
                 'ch_ids': np.arange(j*model_chn_dim, (j+1)*model_chn_dim)}
            batch.append(d)

        # return data and subject indices
        return batch, [d['sid'] for d in batch], inds

    def get_batch(self, i, data, split='train'):
        batch, sid, inds = self._get_batch(i, data, split)
        return batch, sid

    def select_data(self, data, args, split):
        # transform to examples
        data = self.create_examples(data)

        # subsample data
        data = data[::int(1/args.max_trials)]

        # select data based on gpu_id
        num_examples = data.shape[0] // args.num_gpus
        upper = (args.gpu_id + 1) * num_examples
        if args.gpu_id == args.num_gpus - 1:
            upper = data.shape[0]

        data = data[args.gpu_id * num_examples:upper]
        data = torch.IntTensor(data)
        size = data.element_size() * data.nelement() / 1e9

        # check size of data and load to gpu if it's less than 2GB
        if size < 2 or (size < 3 and split == 'train'):
            data = data.to(args.device)
            print(f'Loaded {split} data on gpu {args.gpu_id}')
        else:
            print(f'Loaded {split} data on cpu')

        return data

    def set_common(self, args=None):
        if isinstance(self.args.sample_rate, list):
            w = self.args.sample_rate[1] - self.args.sample_rate[0]
            self.args.sample_rate = w

        args.num_channels = len(args.num_channels)
        args.num_channels = int((args.num_channels-2)/2)

        self.x_train_t = self.select_data(self.x_train_t, args, 'train')
        self.x_val_t = self.select_data(self.x_val_t, args, 'val')
        self.x_test_t = self.select_data(self.x_test_t, args, 'test')

        bs = args.batch_size // args.num_gpus
        self.bs = {'train': bs, 'val': bs, 'test': bs}

        self.train_batches = math.ceil(self.x_train_t.shape[0] / bs)
        self.val_batches = math.ceil(self.x_val_t.shape[0] / bs)
        self.test_batches = math.ceil(self.x_test_t.shape[0] / bs)

        print('Train batches: ', self.train_batches)
        print('Validation batches: ', self.val_batches)
        print('Test batches: ', self.test_batches)

    def create_examples(self, x):
        '''
        Create examples from the continuous data (x).
        '''
        sr = self.args.sample_rate
        shift = sr - self.args.example_shift
        inds = np.arange(x.shape[2] - sr)[::shift]

        x = [x[:, :, ind:ind+sr] for ind in inds]
        x = np.concatenate(x)

        return x

    def load_mat_data(self, args, dtype=np.int16):
        '''
        Loads ready-to-train splits from mat files.
        '''
        chn = args.num_channels
        x_train_ts = []
        x_val_ts = []
        x_test_ts = []

        # load data for each channel
        for index, i in enumerate(chn):
            path = os.path.join(args.load_data, 'ch' + str(i) + '.mat')
            data = loadmat(path)

            x_train_ts.append(np.array(data['x_train_t'], dtype=dtype))
            x_val_ts.append(np.array(data['x_val_t'], dtype=dtype))
            x_test_ts.append(np.array(data['x_test_t'], dtype=dtype))

        self.x_train_t = np.array(x_train_ts).transpose(1, 0, 2)
        self.x_val_t = np.array(x_val_ts).transpose(1, 0, 2)
        self.x_test_t = np.array(x_test_ts).transpose(1, 0, 2)

        path = os.path.join(args.load_data, 'maxabs_scaler')
        self.maxabs = pickle.load(open(path, 'rb'))

    def save_data(self):
        '''
        Save final data to disk for easier loading next time.
        '''
        for i in range(self.x_train_t.shape[0]):
            dump = {'x_train_t': self.x_train_t[i:i+1:, :],
                    'x_val_t': self.x_val_t[i:i+1, :],
                    'x_test_t': self.x_test_t[i:i+1, :]}

            path = os.path.join(self.args.dump_data, 'ch' + str(i) + '.mat')
            savemat(path, dump)

        path = os.path.join(self.args.dump_data, 'maxabs_scaler')
        pickle.dump(self.maxabs, open(path, 'wb'))


class CichyProductQuantized(CichyQuantized):
    def load_data(self, args):
        sr = args.sr_data
        args.sr_data = args.original_sr

        x_trains, x_vals, x_tests = super().load_data(args)
        args.sr_data = sr

        return x_trains, x_vals, x_tests

    def set_common(self, args):
        super().set_common(args)
        self.args.num_channels = self.x_train_t.shape[1] - 2

    def load_mat_data(self, args):
        super().load_mat_data(args, dtype=np.int16)

        self.quantizers = []
        path = os.path.join(args.load_data, 'product_quantizers')
        for i in range(self.args.num_buckets):
            file_path = os.path.join(path, f'product_quantizer_{i}.npz')
            # Load the ResidualQuantizer's parameters
            data = np.load(file_path)

            rq = faiss.ResidualQuantizer(
                int(data['d']), int(data['M']), int(data['nbits']))
            faiss.copy_array_to_vector(data['codebooks'], rq.codebooks)
            rq.is_trained = True
            self.quantizers.append(rq)

        path = os.path.join(args.load_data, 'buckets')
        self.buckets = pickle.load(open(path, 'rb'))

    def save_data(self):
        '''
        Save final data to disk for easier loading next time.
        '''
        self.maxabs = []
        super().save_data()

        path = os.path.join(self.args.dump_data, 'product_quantizers')
        os.makedirs(path, exist_ok=True)
        for i, q in enumerate(self.quantizers):
            file_path = os.path.join(path, f'product_quantizer_{i}.npz')

            cb = faiss.vector_to_array(q.codebooks)
            np.savez(file_path,
                     codebooks=cb,
                     M=2,
                     nbits=self.args.num_bits//2,
                     d=len(self.buckets[i]))

        path = os.path.join(self.args.dump_data, 'buckets')
        pickle.dump(self.buckets, open(path, 'wb'))

    def encode(self):
        # chop last 2 channels
        xtn = self.x_train[:-2, :].T
        xv = self.x_val[:-2, :].T
        xt = self.x_test[:-2, :].T

        xtn, xv, xt = self.product_quantize(xtn, xv, xt)

        xtn = np.append(xtn, self.x_train[-2:, :].astype(np.int16), axis=0)
        xv = np.append(xv, self.x_val[-2:, :].astype(np.int16), axis=0)
        xt = np.append(xt, self.x_test[-2:, :].astype(np.int16), axis=0)

        # resample to 100 Hz
        resample = self.args.original_sr // self.args.sr_data
        self.x_train_t = xtn[:, ::resample]
        self.x_val_t = xv[:, ::resample]
        self.x_test_t = xt[:, ::resample]

    def product_quantize(self, x_train, x_val, x_test):
        # convert to contiguous array
        x_train = np.ascontiguousarray(x_train, dtype=np.float32)
        x_val = np.ascontiguousarray(x_val, dtype=np.float32)
        x_test = np.ascontiguousarray(x_test, dtype=np.float32)

        # put the 306 channels into 30 buckets based on covariance
        # in each bucket the channels should have similar covariances
        num_buckets = self.args.num_buckets

        # Compute the covariance matrix of the features
        cov_matrix = np.cov(x_train, rowvar=False)

        # Apply K-means clustering on the covariance matrix
        kmeans = KMeans(n_clusters=num_buckets, random_state=0).fit(cov_matrix)

        # Create a dictionary to store the features in each bucket
        buckets = {i: [] for i in range(num_buckets)}

        # Assign features to the corresponding buckets
        for feature_idx, bucket in enumerate(kmeans.labels_):
            buckets[bucket].append(feature_idx)

        quantizers = []
        xtrains = []
        xvals = []
        xtests = []
        # create a residual quantizer for each bucket of features
        for bucket, features in buckets.items():
            res_quant = faiss.ResidualQuantizer(len(features),
                                                2,
                                                self.args.num_bits//2)
            res_quant.train(np.ascontiguousarray(x_train[:, features]))
            quantizers.append(res_quant)

            xt = self.compute_codes(res_quant, x_train[:, features])
            xtrains.append(xt)

            xv = self.compute_codes(res_quant, x_val[:, features])
            xvals.append(xv)

            xt = self.compute_codes(res_quant, x_test[:, features])
            xtests.append(xt)

        self.quantizers = quantizers
        self.buckets = buckets

        recon = self.reconstruct(np.array(xvals))
        err = ((x_val - recon)**2).sum() / (x_val ** 2).sum()
        print('Product quantization reconstruction error: ', err)

        return np.array(xtrains), np.array(xvals), np.array(xtests)

    def compute_codes(self, quantizer, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        codes = quantizer.compute_codes(x).astype(np.int16)

        # flatten vector with 256 x 128 unique values into a single vocab
        multiplier = 2**self.args.num_bits // 256
        codes = codes[:, 0] * multiplier + codes[:, 1]
        return codes

    def reconstruct(self, x):
        # reconstruct the 306 channels from the 30 buckets
        x = np.ascontiguousarray(x)

        # invert vocab to 256 x 128 unique values
        divider = 2**self.args.num_bits // 256
        codes = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.uint8)
        codes[:, :, 1] = x % divider
        codes[:, :, 0] = x // divider

        # reconstruct the 306 channels from the 30 buckets
        num_channels = sum([len(feats) for feats in self.buckets.values()])
        x_recon = np.zeros((x.shape[1], num_channels))
        for bucket, feats in self.buckets.items():
            x_recon[:, feats] = self.quantizers[bucket].decode(codes[bucket])

        return x_recon


class CichyQuantizedRandomCond(CichyQuantized):
    def __init__(self, args):
        super().__init__(args)

        cond_path = os.path.join(args.data_dir, 'cond_labels.npy')
        # try loading condition labels from disk
        if os.path.exists(cond_path):
            self.cond_labels = np.load(cond_path)
            self.cond_labels = torch.Tensor(self.cond_labels).cuda().long()
            return

        # create fake condition labels
        data_length = self.x_train_t.shape[2]//2 * self.x_train_t.shape[0]
        seconds = data_length//self.args.sr_data
        epoch_len = self.args.sr_data//2
        cond = []
        for s in range(seconds):
            # choose a class randomly from self.args.num_classes
            cl = np.random.randint(1, self.args.num_classes)
            cond.append(np.array([cl]*epoch_len))

            # uniform distribution between 0.9 and 1
            num_zeros = np.random.randint(int(epoch_len*0.8), epoch_len)
            cond.append(np.zeros((num_zeros)))

        shift = None
        gen_len = None
        train = None
        cond = np.concatenate(cond)[:shift+gen_len]
        # replace first epoch with train cond channel
        cond = torch.Tensor(cond).cuda().long()
        cond[:shift] = train[0, -2, :shift]

    def get_batch(self, i, data, split):
        '''
        Get batch of data.
        '''
        batch, sid, inds = super()._get_batch(i, data, split)

        # replace cond field of batch with self.cond_labels
        for i in len(batch):
            batch[i]['condition'] = self.cond_labels[inds]

        return batch, sid
    

class CichyQuantizedRandomLabel(CichyQuantized):
    def random_label(self, x):
        cond = x[:, -2, :]

        for i in range(cond.shape[0]):
            # get the unique labels in this epoch, except 0
            labels = np.unique(cond[i].cpu().numpy())
            labels = labels[labels != 0]

            # iterate over unique labels
            for l in labels:
                # replace all instances of this label with a new label
                new_label = torch.randint(1, self.args.num_classes, (1,))
                cond[i, cond[i] == l] = new_label

        return cond

    def set_common(self, args=None):
        super().set_common(args)

        # randomize the condition labels of self.x_train_t
        self.x_train_t[:, -2, :] = self.random_label(self.x_train_t)
        self.x_val_t[:, -2, :] = self.random_label(self.x_val_t)
        self.x_test_t[:, -2, :] = self.random_label(self.x_test_t)

    def _get_batch_testing(self, i, data, split):
        '''
        Get batch of data.
        '''
        batch, sid, inds = super()._get_batch(i, data, split)

        # replace cond field of batch with self.cond_labels
        for i, b in enumerate(batch):
            # get the unique labels in this batch, except 0
            labels = np.unique(b['condition'].cpu().numpy())
            labels = labels[labels != 0]

            # iterate over unique labels
            for l in labels:
                # replace all instances of this label with a new label
                new_label = np.random.randint(1, self.args.num_classes)
                batch[i]['condition'][b['condition'] == l] = new_label

        return batch, sid


class CichyQuantizedSimulation(CichyQuantized):
    def load_data(self, args):
        '''
        Load raw data from multiple subjects.
        '''
        # whether we are working with one subject or a directory of them
        if isinstance(args.data_path, list):
            paths = args.data_path
        else:
            paths = os.listdir(args.data_path)
            paths = [p for p in paths if 'sub' in p]
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if not os.path.isdir(p)]
        print('Number of subjects: ', len(paths))

        resample = int(args.original_sr/args.sr_data)

        x_trains = []
        x_vals = []
        x_tests = []
        for sid, path in enumerate(paths):
            print(path)
            dataset = np.load(path)

            # choose first 306 channels and downsample
            dataset = dataset[args.num_channels, ::resample]
            labels = [0] * dataset.shape[1]

            labels = np.array(labels)

            split_v = int(dataset.shape[1]*0.13)
            split_t = int(dataset.shape[1]*0.26)
            labels = {'val': labels[:split_v].reshape(1, -1),
                      'test': labels[split_v:split_t].reshape(1, -1),
                      'train': labels[split_t:].reshape(1, -1)}

            # create training and validation splits
            x_val = dataset[:, :split_v]
            x_test = dataset[:, split_v:split_t]
            x_train = dataset[:, split_t:]

            x_train, x_val, x_test = self.normalize(x_train, x_val, x_test)

            # add labels to data
            subid = np.array([sid] * x_val.shape[1]).reshape(1, -1)
            x_val = np.concatenate((x_val, labels['val'], subid), axis=0)

            subid = np.array([sid] * x_test.shape[1]).reshape(1, -1)
            x_test = np.concatenate((x_test, labels['test'], subid), axis=0)

            subid = np.array([sid] * x_train.shape[1]).reshape(1, -1)
            x_train = np.concatenate((x_train, labels['train'], subid), axis=0)

            x_trains.append(x_train)
            x_tests.append(x_test)
            x_vals.append(x_val)

        return x_trains, x_vals, x_tests


class CichyQuantizedBatched(CichyQuantized):
    def __init__(self, args):
        '''
        Load data and apply pca, then create batches.
        '''
        self.args = args
        self.inds = {'train': [], 'val': [], 'test': []}
        self.sub_id = {'train': [0], 'val': [0], 'test': [0]}
        self.chn_weights_sample = {}
        self.chn_ids = {}

        if isinstance(self.args.sample_rate, list):
            w = self.args.sample_rate[1] - self.args.sample_rate[0]
            self.args.sample_rate = w

        # load data created by CichyQuantized and save individual examples
        if args.load_data != args.dump_data:
            super().load_mat_data(args)

            args.num_channels = len(args.num_channels)
            args.num_channels = int((args.num_channels-2)/2)

            # transform to examples
            self.x_train_t = self.create_examples(self.x_train_t)
            self.x_val_t = self.create_examples(self.x_val_t)
            self.x_test_t = self.create_examples(self.x_test_t)

            # save examples
            self.save_data()

            return

        self.set_common(args)

        # dummy data
        self.x_train_t = None
        self.x_val_t = None
        self.x_test_t = None

    def _get_batch(self, i, data, split):
        '''
        Get batch of data.
        '''
        num_chn = self.args.num_channels

        if i == 0:
            self.inds[split] = np.random.permutation(self.ex[split])

        if len(self.inds[split]) > self.bs[split]:
            bs = self.bs[split]
        else:
            bs = len(self.inds[split])

        # sample random indices
        inds = self.inds[split][:bs]

        # read examples from disk according to correct split
        # parallelize the for loop to increase speed
        data = []
        for j in inds:
            path = os.path.join(self.args.load_data, split + str(j) + '.npy')
            data.append(torch.Tensor(np.load(path)).long())
        
        # stack individual examples to form a batch
        data = torch.stack(data)
        data = data.cuda()

        # remove the already sampled indices
        self.inds[split] = self.inds[split][bs:]
        
        # data: 306 input chs, 306 target chns, 1 condition id, 1 subject id
        data = {'inputs': data[:, :num_chn, :],
                'targets': data[:, num_chn:num_chn*2, :],
                'condition': data[:, -2:-1, :],
                'sid': data[:, -1:, :]}

        # return data and subject indices
        return data, data['sid']

    def get_batch(self, i, data, split):
        '''
        Get batch of data.
        '''
        if not self.args.class_mode:
            return self._get_batch(i, data, split)

        sr = self.args.sample_rate
        data = []
        for j in range(self.ex[split]):
            path = os.path.join(self.args.load_data, split + str(j) + '.npy')
            data.append(np.load(path)[:, :sr//2])

        data = np.concatenate(data, axis=1)
        data = torch.Tensor(data).long().cuda()

        # get indices where condition turns on
        diff = data[-2, 1:] - data[-2, :-1]
        inds = (diff > 0).nonzero().squeeze()

        baseline = self.args.sr_data // 10
        rf = self.args.rf
        # select examples around where inds is in the middle
        data = [data[:, i-rf-baseline:i-rf-baseline+sr] for i in inds]
        data = torch.stack(data)

        num_chn = self.args.num_channels
        # data: 306 input chs, 306 target chns, 1 condition id, 1 subject id
        data = {'inputs': data[:, :num_chn, :],
                'targets': data[:, num_chn:num_chn*2, :],
                'condition': data[:, -2:-1, :],
                'sid': data[:, -1:, :]}

        return data, data['sid']

    def set_examples(self, split):
        path = self.args.dump_data
        exs = [name for name in os.listdir(path) if split in name]
        self.ex[split] = len(exs)

    def set_common(self, args=None):
        if isinstance(self.args.sample_rate, list):
            w = self.args.sample_rate[1] - self.args.sample_rate[0]
            self.args.sample_rate = w

        args.num_channels = len(args.num_channels)
        args.num_channels = int((args.num_channels-2)/2)

        bs = args.batch_size
        self.bs = {'train': bs, 'val': bs, 'test': bs}

        # count files with train in their name in args.dump_data
        self.ex = {}
        self.set_examples('train')
        self.set_examples('val')
        self.set_examples('test')

        self.train_batches = math.ceil(self.ex['train'] / self.bs['train'])
        self.val_batches = math.ceil(self.ex['val'] / self.bs['val'])
        self.test_batches = math.ceil(self.ex['test'] / self.bs['test'])

        print('Train batches: ', self.train_batches)
        print('Validation batches: ', self.val_batches)
        print('Test batches: ', self.test_batches)

    def save_data(self):
        '''
        Save each example from self.x separately to disk
        '''

        # check if dump_data directory exists
        if not os.path.exists(self.args.dump_data):
            os.makedirs(self.args.dump_data)

        # save train data as numpy arrays
        for i in range(self.x_train_t.shape[0]):
            path = os.path.join(self.args.dump_data, 'train' + str(i))
            np.save(path, self.x_train_t[i])

        # save validation data as numpy arrays
        for i in range(self.x_val_t.shape[0]):
            path = os.path.join(self.args.dump_data, 'val' + str(i))
            np.save(path, self.x_val_t[i])

        # save test data as numpy arrays
        for i in range(self.x_test_t.shape[0]):
            path = os.path.join(self.args.dump_data, 'test' + str(i))
            np.save(path, self.x_test_t[i])

    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        '''
        pass


class CichyQuantizedAR(CichyQuantized):
    def save_data(self):
        '''
        Save final data to disk for easier loading next time.
        '''
        for i in range(self.x_train.shape[0]):
            dump = {'x_train_t': self.x_train[i:i+1:, :],
                    'x_val_t': self.x_val[i:i+1, :],
                    'x_test_t': self.x_test[i:i+1, :]}

            path = os.path.join(self.args.dump_data, 'ch' + str(i) + '.mat')
            savemat(path, dump)

        path = os.path.join(self.args.dump_data, 'maxabs_scaler')
        pickle.dump(self.maxabs, open(path, 'wb'))

    def set_common(self, args=None):
        if isinstance(self.args.sample_rate, list):
            w = self.args.sample_rate[1] - self.args.sample_rate[0]
            self.args.sample_rate = w

        args.num_channels = len(args.num_channels)

        # transform to examples
        self.x_train_t = self.create_examples(self.x_train_t)
        self.x_val_t = self.create_examples(self.x_val_t)
        self.x_test_t = self.create_examples(self.x_test_t)

        try:
            self.x_train_t = torch.Tensor(self.x_train_t).cuda()
            self.x_val_t = torch.Tensor(self.x_val_t).cuda()
            self.x_test_t = torch.Tensor(self.x_test_t).cuda()
            print('Data loaded on gpu.')
        except RuntimeError:
            self.x_train_t = torch.Tensor(self.x_train_t)
            self.x_val_t = torch.Tensor(self.x_val_t)
            self.x_test_t = torch.Tensor(self.x_test_t)
            print('Data loaded on cpu.')

        bs = args.batch_size
        self.bs = {'train': bs, 'val': bs, 'test': bs}

        self.train_batches = math.ceil(
            self.x_train_t.shape[0] / self.bs['train']) 
        self.val_batches = math.ceil(self.x_val_t.shape[0] / self.bs['val'])
        self.test_batches = math.ceil(self.x_test_t.shape[0] / self.bs['test'])

        print('Train batches: ', self.train_batches)
        print('Validation batches: ', self.val_batches)
        print('Test batches: ', self.test_batches)

    def get_batch(self, i, data, split):
        '''
        Get batch of data.
        '''
        num_chn = self.args.num_channels

        if i == 0:
            self.inds[split] = np.random.permutation(data.shape[0])

        if len(self.inds[split]) > self.bs[split]:
            bs = self.bs[split]
        else:
            bs = len(self.inds[split])

        # sample random indices
        inds = self.inds[split][:bs]

        if data.is_cuda:
            data = data[inds]
        else:
            data = data[inds].cuda()

        # remove the already sampled indices
        self.inds[split] = self.inds[split][bs:]
        
        # data: 306 input chs, 306 target chns, 1 condition id, 1 subject id
        data = {'inputs': data[:, :, :-1],
                'targets': data[:, :, 1:]}

        # add encoding functions to the data dict
        data['maxabs'] = self.maxabs

        # return data and subject indices
        return data, None


class CichyQuantizedARSimulation(CichyQuantizedAR, CichyQuantizedSimulation):
    pass


class CichyQuantizedGauss(CichyQuantized):
    def get_batch(self, i, data, split):
        '''
        Get batch of data.
        '''
        data, sid = super().get_batch(i, data, split)

        data['gauss'] = self.gauss_targets

        return data, sid

    def gauss_filter(self, x):
        '''
        Gaussian filter for the continuous data.
        '''
        nc = self.args.num_channels
        targets = x[:, nc:2*nc, :]
        targets = torch.nn.functional.one_hot(targets)

        targets = targets.float().cpu().numpy()
        targets = gaussian_filter1d(targets, sigma=2)

        return targets

    def set_common(self, args):
        super().set_common(args)

        # create a tensor of one-hot gauss targets
        self.gauss_targets = gaussian_filter1d(np.eye(args.mu+1),
                                               sigma=2)
        self.gauss_targets = torch.Tensor(self.gauss_targets).cuda()


class CichyQuantizedDecoding(CichyQuantized):
    def create_examples(self, x):
        '''
        Create examples from the continuous data (x).
        '''
        rf = self.args.rf
        nc = self.args.num_channels

        channel_inds = list(range(nc)) + [2*nc, 2*nc+1]

        # find the indices of the start of each example based on -100ms
        # before the stimulus in the condition channel (612)
        shifted = x[:, -2, 1:] - x[:, -2, :-1]
        inds = np.where(shifted > 0)[0] - int(self.args.sr_data/10) + 1

        x = [x[:, channel_inds, ind:ind+rf] for ind in inds]
        x = np.concatenate(x)

        return x


class CichyDataCPU(CichyData):
    '''
    Convienence class where all data is on the CPU.
    '''
    def set_common(self):
        # set common parameters
        self.bs = {'train': self.args.batch_size,
                   'val': int(self.args.batch_size*self.args.split)}

        self.train_batches = int(self.x_train_t.shape[0] / self.bs['train'])
        self.val_batches = int(self.x_val_t.shape[0] / self.bs['val'])

        self.x_train_t = torch.Tensor(self.x_train_t).float()
        self.x_val_t = torch.Tensor(self.x_val_t).float()
        self.sub_id['train'] = torch.LongTensor(self.sub_id['train']).cuda()
        self.sub_id['val'] = torch.LongTensor(self.sub_id['val']).cuda()
        self.sub_id['train'] = self.sub_id['train'].reshape(-1)
        self.sub_id['val'] = self.sub_id['val'].reshape(-1)


class MRCDataCPU(MRCData):
    def set_common(self):
        CichyDataCPU.set_common(self)


class CichyCombinedData:
    '''
    Dataset for training simultaneously on epoched and continuous data.
    This is used in conjuction with a model using both
    classification and forecasting losses.
    '''
    def __init__(self, args):
        combined = args.load_data
        bs = args.batch_size

        # create separate datasets for the 2 problems (continuous and epoched)
        # data is stored on CPU because of memory limitations
        args.load_data = combined['epoched']
        args.batch_size = bs['epoched']
        self.epoched = CichyDataCPU(args)

        args.batch_size = bs['cont']
        args.load_data = combined['cont']
        args.num_channels = list(range(306))
        self.cont = MRCDataCPU(args)

        self.train_batches = min(self.cont.train_batches, self.epoched.train_batches)
        self.val_batches = min(self.cont.val_batches, self.epoched.val_batches)

        print(self.cont.train_batches)
        print(self.epoched.train_batches)

    def get_train_batch(self, i):
        # helper for getting a training batch from both datasets
        epoched_dat, epoched_sid = self.epoched.get_train_batch(i)
        cont_dat, cont_sid = self.cont.get_train_batch(i)

        # this is where data is put on the GPU
        ret_dat = {'epoched': epoched_dat.cuda(), 'cont': cont_dat.cuda()}
        ret_sid = {'epoched': epoched_sid, 'cont': cont_sid}

        return ret_dat, ret_sid

    def get_val_batch(self, i):
        # helper for getting a validation batch from both datasets
        epoched_dat, epoched_sid = self.epoched.get_val_batch(i)
        cont_dat, cont_sid = self.cont.get_val_batch(i)

        # this is where data is put on the GPU
        ret_dat = {'epoched': epoched_dat.cuda(), 'cont': cont_dat.cuda()}
        ret_sid = {'epoched': epoched_sid, 'cont': cont_sid}

        return ret_dat, ret_sid


class CichyDataAttention(CichyData):
    # this is not used
    def __init__(self, args):
        super(CichyDataAttention, self).__init__(args)
        self.x_train_t = self.x_train_t.permute(2, 0, 1)
        self.x_val = self.x_val_t.permute(2, 0, 1)
