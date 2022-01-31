import os
import sys
import torch
import random
import numpy as np

from scipy.io import loadmat, savemat

from mrc_data import MRCData


class CichyData(MRCData):
    '''
    Class for loading the trials from the Cichy dataset.
    '''
    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        '''
        chn = args.num_channels
        num_ch = len(chn) - 1
        x_train_ts = []
        x_val_ts = []

        # load data for each channel
        for index, i in enumerate(chn):
            data = loadmat(args.load_data + 'ch' + str(i) + '.mat')
            x_train_ts.append(np.array(data['x_train_t']))
            x_val_ts.append(np.array(data['x_val_t']))

            if index == 0:
                self.sub_id['train'] = np.array(data['sub_id_train'])
                self.sub_id['val'] = np.array(data['sub_id_val'])

        self.x_train_t = np.concatenate(tuple(x_train_ts), axis=1)
        self.x_val_t = np.concatenate(tuple(x_val_ts), axis=1)

        # crop data
        #tmin = args.sample_rate[0]
        #tmax = args.sample_rate[1]
        self.x_train_t = self.x_train_t[:, :, :args.sample_rate]
        self.x_val_t = self.x_val_t[:, :, :args.sample_rate]

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
            x_train = x_train.reshape(-1, num_ch)
            x_val = x_val.reshape(-1, num_ch)

            x_train, x_val = self.whiten(x_train, x_val)

            # reshape back to trials
            x_train = x_train.reshape(-1, self.args.sample_rate, num_ch)
            x_val = x_val.reshape(-1, self.args.sample_rate, num_ch)
            x_train = x_train.transpose(0, 2, 1)
            x_val = x_val.transpose(0, 2, 1)

            self.x_train_t[:, :num_ch, :] = x_train
            self.x_val_t[:, :num_ch, :] = x_val

        args.num_channels = args.num_channels[:-1]

    def save_data(self):
        '''
        Save final data to disk for easier loading next time.
        '''
        for i in range(self.x_train_t.shape[1]):
            dump = {'x_train_t': self.x_train_t[:, i:i+1:, :],
                    'x_val_t': self.x_val_t[:, i:i+1, :],
                    'sub_id_train': self.sub_id['train'],
                    'sub_id_val': self.sub_id['val']}
            savemat(self.args.dump_data + 'ch' + str(i) + '.mat', dump)

    def load_data(self, args):
        '''
        Load trials for each condition from multiple subjects.
        '''
        # whether we are working with one subject or a directory of them
        if 'sub' in args.data_path:
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
        for path in paths:
            print('Loading ', path)
            min_trials = 100
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
            print('Minimum trials: ', min_trials)

            # dataset shape: conditions x trials x channels x timesteps
            dataset = np.array([t[:min_trials, :, :] for t in dataset])

            # choose first 306 channels
            dataset = dataset.transpose(0, 1, 3, 2)
            dataset = dataset[:, :, args.num_channels, :]
            self.timesteps = dataset.shape[3]

            # create training and validation splits with equal class numbers
            split = int(args.split * min_trials)
            x_val = dataset[:, :split, :, :]
            x_train = dataset[:, split:, :, :]

            x_train = x_train.transpose(0, 1, 3, 2).reshape(-1, channels)
            x_val = x_val.transpose(0, 1, 3, 2).reshape(-1, channels)

            # standardize dataset along channels
            x_train, x_val = self.normalize(x_train, x_val)

            x_trains.append(x_train)
            x_vals.append(x_val)

        # this is just needed to work together with other dataset classes
        disconts = [[0] for path in paths]
        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, disconts

    def create_examples(self, x, disconts):
        '''
        Create examples with labels.
        '''

        # expand shape to trials
        x = x.transpose(1, 0)
        x = x.reshape(self.args.num_classes, -1, self.timesteps, x.shape[1])
        x = x.transpose(0, 1, 3, 2)

        # downsample data if needed
        resample = int(1000/self.args.sr_data)
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


class CichyContData(MRCData):
    '''
    Implements the continuous classification problem on the Cichy dataset.
    Under construction.
    '''
    def load_data(self, args):
        '''
        Load raw data from multiple subjects.
        '''
        if '.mat' in args.data_path:
            paths = [args.data_path]
        else:
            paths = os.listdir(args.data_path)
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if not os.path.isdir(p)]
            paths = [p for p in paths if 'subject' in p.split('/')[-1]]
        print('Number of subjects: ', len(paths))

        resample = int(1000/args.sr_data)
        epoch_len = int(0.5*args.sr_data)

        x_trains = []
        x_vals = []
        disconts = []
        for path in paths:
            print(path)

            # will have to be changed to handle concatenated subjects
            ev_path = os.path.join(args.data_path, 'event_times.npy')
            event_times = [(int(ev[0]/resample), ev[2]) for ev in np.load(ev_path)]

            x_train = np.load(path).T
            d = np.array([0])

            for i, val in enumerate(d[1:]):
                d[i+1] = d[i] + val
            disconts.append(list((d/resample).astype(int)))

            # choose first 306 channels and downsample
            x_train = x_train[args.num_channels, ::resample]
            labels = [0] * x_train.shape[1]

            val_counter = [0] * args.num_classes
            val_events = []
            train_events = []
            # set labels
            for ev in event_times:
                if ev[1] < 119:
                    labels[ev[0]:ev[0]+epoch_len] = [ev[1]] * epoch_len

                    if val_counter[ev[1]-1] < 4:
                        val_counter[ev[1]-1] += 1
                        val_events.append(ev[0])
                    else:
                        train_events.append(ev[0])

            labels = np.array(labels)

            split = int((max(val_events) + min(train_events))/2)
            labels = {'val': labels[:split].reshape(1, -1),
                      'train': labels[split:].reshape(1, -1)}

            # create training and validation splits
            x_val = x_train[:, :split]
            x_train = x_train[:, split:]

            x_train, mean, var = self.normalize(x_train)
            x_val, _, _ = self.normalize(x_val, mean, var)

            # add labels to data
            x_val = np.concatenate((x_val, labels['val']), axis=0)
            x_train = np.concatenate((x_train, labels['train']), axis=0)

            x_trains.append(x_train)
            x_vals.append(x_val)

        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, disconts

    def create_examples(self, x, disconts):
        '''
        Create examples with labels.
        '''
        return x.reshape(1, x.shape[0], x.shape[1])

    def set_common(self):
        # set common parameters
        super(CichyContData, self).set_common()

        self.train_batches = int((self.x_train_t.shape[1] - self.args.rf - 1) / self.bs['train'])
        self.val_batches = int((self.x_val_t.shape[1] - self.args.rf - 1) / self.bs['val'])

    def get_batch(self, i, data, split):
        rf = self.args.rf
        if i == 0:
            self.inds[split] = list(range(data.shape[1] - rf))

        # sample random indices
        inds = random.sample(self.inds[split], self.bs[split])
        self.inds[split] = [v for v in self.inds[split] if v not in inds]

        data = torch.stack([data[0, :, ind:ind+rf] for ind in inds])

        return data, self.sub_id[split]


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
