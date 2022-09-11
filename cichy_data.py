import os
import sys
import torch
import random
import pickle
import mne
import traceback
import numpy as np

from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA

from mrc_data import MRCData


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

        # select a subset of training trials
        num_trials = np.sum(self.x_train_t[:, num_ch, 0] == 0.0)
        max_trials = int(args.max_trials * num_trials)
        trials = [0] * args.num_classes

        inds = []
        for i in range(self.x_train_t.shape[0]):
            cond = int(self.x_train_t[i, num_ch, 0])
            if trials[cond] < max_trials:
                trials[cond] += 1
                inds.append(i)

        self.x_train_t = self.x_train_t[inds, :, :]

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

            # loop over 118 conditions
            for c in range(args.num_classes):
                cond_path = os.path.join(path, 'cond' + str(c))
                files = os.listdir(cond_path)
                files = [f for f in files if 'npy' in f]
                if len(files) < min_trials:
                    min_trials = len(files)

                trials = []
                # loop over trials within a condition
                for f in files:
                    trial = np.load(os.path.join(cond_path, f))
                    trials.append(trial)

                dataset.append(np.array(trials))

            # condition with lowest number of trials
            print('Minimum trials: ', min_trials, flush=True)

            # dataset shape: conditions x trials x timesteps x channels
            dataset = np.array([t[:min_trials, :, :] for t in dataset])

            # choose first 306 channels
            dataset = dataset.transpose(0, 1, 3, 2)
            dataset = dataset[:, :, args.num_channels, :]
            self.timesteps = dataset.shape[3]

            # create training and validation splits with equal class numbers
            x_train, x_val, x_test = self.splitting(dataset, args)

            # crop training trials
            max_trials = round(args.max_trials * x_train.shape[1])
            x_train = x_train[:, :max_trials, :, :]

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


class CichyDataCrossval(CichyData):
    def splitting(self, dataset, args):
        split = args.split[1] - args.split[0]
        split = int(split*dataset.shape[1])

        for i in range(dataset.shape[0]):
            perm = np.random.permutation(dataset.shape[1])
            dataset[i, :, :, :] = dataset[i, perm, :, :]

        # create separate val and test splits
        x_val = dataset[:, :split, :, :]
        x_train = dataset[:, split:, :, :]
        x_test = x_train[:, :split:, :, :]
        x_train = x_train[:, split:, :, :]

        return x_train, x_val, x_test


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
        x_trains = np.concatenate(tuple(x_trains), axis=1)
        x_vals = np.concatenate(tuple(x_vals), axis=1)
        x_tests = np.concatenate(tuple(x_tests), axis=1)

        xtn, xv, xtt = self.encode(x_trains[:-2], x_vals[:-2], x_tests[:-2])

        # append back labels and sid
        self.x_train_t = np.append(xtn, x_trains[-2:, :-1], axis=0)
        self.x_val_t = np.append(xv, x_vals[-2:, :-1], axis=0)
        self.x_test_t = np.append(xtt, x_tests[-2:, :-1], axis=0)

        if not os.path.isdir(os.path.split(args.dump_data)[0]):
            os.mkdir(os.path.split(args.dump_data)[0])

        self.save_data()
        self.set_common(args)

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
            ev_path = os.path.join(args.data_path, 'event_times.npy')
            event_times = np.load(ev_path)
            event_times = [(int(ev[0]/resample), ev[2]) for ev in event_times]

            dataset = np.load(path).T

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

    def clip(self, x):
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
        mu = self.args.mu
        shape = x.shape

        x = x.reshape(-1)
        x = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)

        digitized = ((x + 1) / 2 * mu+0.5).astype(np.int32)
        x = 2 * ((digitized).astype(np.float32) / mu) - 1

        x = x.reshape(shape)
        digitized = digitized.reshape(shape)

        x = np.append(x[:, :-1], digitized[:, 1:], axis=0)
        return x

    def encode(self, xtn, xv, xtt):
        self.pca = PCA(self.args.whiten)
        self.maxabs = MaxAbsScaler()

        xtn = self.pca.fit_transform(xtn.T)
        xv = self.pca.transform(xv.T)
        xtt = self.pca.transform(xtt.T)

        xtn = self.clip(xtn.T).T

        xtn = self.maxabs.fit_transform(xtn).T
        xv = self.maxabs.transform(xv).T
        xtt = self.maxabs.transform(xtt).T

        xv = np.clip(xv, -1, 1)
        xtt = np.clip(xtt, -1, 1)

        return self.mulaw(xtn), self.mulaw(xv), self.mulaw(xtt)

    def decode(self, x):
        pass
        '''

        x = x / (mu + 1) * 2 - 1
        x = np.sign(x)*((mu+1)**np.abs(x)-1) / mu

        x = x.reshape(shape)
        x = maxabs.inverse_transform(x.T)
        x = pca.inverse_transform(x)
        x = robust.inverse_transform(x)

        return x.T
        '''

    def get_batch(self, i, data, split):
        num_chn = self.args.num_channels
        sr = self.args.sample_rate

        if i == 0:
            self.inds[split] = np.random.permutation(data.shape[0])

            chn_ids = torch.multinomial(
                self.chn_weights, data.shape[0], replacement=True)
            chn_ids = chn_ids.reshape(-1, 1, 1)

            chn_weights_sample = self.chn_weights[chn_ids].reshape(-1, 1, 1)

            # reshape to concatenate with data
            self.chn_ids[split] = torch.repeat_interleave(chn_ids, sr, dim=2)
            self.chn_weights_sample[split] = torch.repeat_interleave(
                chn_weights_sample, sr, dim=2)

        # sample random indices
        inds = self.inds[split][:self.bs[split]]
        data = data[inds]

        # remove the already sampled indices
        self.inds[split] = self.inds[split][self.bs[split]:]

        # data: 306 input chs, 306 target chns, 1 condition id, 1 subject id,
        # 1 channel id, 1 channel weight
        data = {'inputs': data[:, :num_chn, :],
                'targets': data[:, num_chn:num_chn*2, :].long(),
                'condition': data[:, -2:-1, :].long(),
                'sid': data[:, -1:, :].long(),
                'chnid': self.chn_ids[split][:self.bs[split]],
                'chn_weights': self.chn_weights_sample[split][:self.bs[split]]}

        # remove the already sampled indices
        self.chn_ids[split] = self.chn_ids[split][self.bs[split]:]
        self.chn_weights_sample[split] = self.chn_weights_sample[split][self.bs[split]:]

        # return data and subject indices
        return data, data['sid']

    def set_common(self, args=None):
        if isinstance(self.args.sample_rate, list):
            w = self.args.sample_rate[1] - self.args.sample_rate[0]
            self.args.sample_rate = w

        # transform to examples
        self.x_train_t = self.create_examples(self.x_train_t)
        self.x_val_t = self.create_examples(self.x_val_t)
        self.x_test_t = self.create_examples(self.x_test_t)

        # only use subset of data
        sampling = int(1/args.max_trials)
        self.x_train_t = self.crop_trials(self.x_train_t, sampling)
        self.x_val_t = self.crop_trials(self.x_val_t, sampling)
        self.x_test_t = self.crop_trials(self.x_test_t, sampling)

        super(CichyQuantized, self).set_common(args)

        args.num_channels = args.whiten

        try:
            self.chn_weights = torch.Tensor(self.chn_weights).float().cuda()
        except Exception:
            traceback.print_exc()

    def crop_trials(self, data, sampling):
        inds = np.arange(data.shape[0])[::sampling]
        return data[inds]

    def create_examples(self, x):
        '''
        Create examples from the continuous data (x).
        '''
        sr = self.args.sample_rate
        inds = np.arange(x.shape[2] - sr)[::int(sr/2)]

        x = [x[:, :, ind:ind+sr] for ind in inds]
        x = np.concatenate(x)

        #x = np.array(np.split(inds[1:]))

        return x

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
            path = os.path.join(args.load_data, 'ch' + str(i) + '.mat')
            data = loadmat(path)

            x_train_ts.append(np.array(data['x_train_t']))
            x_val_ts.append(np.array(data['x_val_t']))
            x_test_ts.append(np.array(data['x_test_t']))

        self.x_train_t = np.array(x_train_ts).transpose(1, 0, 2)
        self.x_val_t = np.array(x_val_ts).transpose(1, 0, 2)
        self.x_test_t = np.array(x_test_ts).transpose(1, 0, 2)

        path = os.path.join(args.load_data, 'pca_model')
        self.pca = pickle.load(open(path, 'rb'))

        path = os.path.join(args.load_data, 'maxabs_scaler')
        self.maxabs = pickle.load(open(path, 'rb'))

        self.chn_weights = torch.tensor(self.pca.explained_variance_ratio_)
        #print(self.chn_weights)

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

        path = os.path.join(self.args.dump_data, 'pca_model')
        pickle.dump(self.pca, open(path, 'wb'))

        path = os.path.join(self.args.dump_data, 'maxabs_scaler')
        pickle.dump(self.maxabs, open(path, 'wb'))


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
