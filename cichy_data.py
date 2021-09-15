import os
import random
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt

from mrc_data import MRCData


class CichyData(MRCData):
    '''
    Class for loading the trials from the Cichy dataset.
    '''
    def filter(self, x):
        b, a = butter(5, [0.1, 124.99], 'bandpass', fs=1000)
        x = filtfilt(b, a, x)

        b, a = butter(5, [49.8, 50.2], 'bandstop', fs=1000)
        x = filtfilt(b, a, x)

        b, a = butter(5, [99.7, 100.3], 'bandstop', fs=1000)
        x = filtfilt(b, a, x)

        b, a = butter(5, [149.6, 150.4], 'bandstop', fs=1000)
        x = filtfilt(b, a, x)

        return x

    def normalize(self, x, mean=None, var=None):
        '''
        Normalize along channels, by concatenating all trials.
        '''
        channels = x.shape[2]
        x = x.transpose(0, 1, 3, 2).reshape(-1, channels).transpose(1, 0)
        x, mean, var = super(CichyData, self).normalize(x, mean, var)

        return x, mean, var

    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        '''
        chn = args.num_channels
        args.num_channels = args.num_channels[:-1]
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

        # crop data according to receptive field
        self.x_train_t = self.x_train_t[:, :, :args.sample_rate]
        self.x_val_t = self.x_val_t[:, :, :args.sample_rate]

        print(self.x_train_t.shape[0])
        print(self.x_val_t.shape[0])

        # shuffle labels
        #self.x_train_t[:, -1, 0] = np.random.randint(0, 118, (self.x_train_t.shape[0],))

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

        x_trains = []
        x_vals = []
        disconts = []
        for path in paths:
            # store condition with lowest number of trials
            min_trials = 100
            dataset = []
            # loop over 118 conditions
            for c in range(0, 118):
                cond_path = os.path.join(path, 'cond' + str(c))
                files = os.listdir(cond_path)
                if len(files) < min_trials:
                    min_trials = len(files)

                trials = []
                for f in files:
                    #trial = loadmat(os.path.join(cond_path, f))
                    #trials.append(np.array(trial['F']))
                    trial = np.load(os.path.join(cond_path, f))
                    trials.append(trial)

                dataset.append(np.array(trials))

            print('Minimum trials: ', min_trials)
            # dataset shape: conditions x trials x channels x timesteps
            dataset = np.array([t[:min_trials, :, :] for t in dataset])

            # choose first 306 channels and downsample
            dataset = dataset.transpose(0, 1, 3, 2)
            dataset = dataset[:, :, args.num_channels, :]
            self.timesteps = dataset.shape[3]

            #dataset = self.filter(dataset)

            # shuffle trials
            shuffled = list(range(min_trials))
            random.shuffle(shuffled)
            dataset = dataset[:, shuffled, :, :]

            # create training and validation splits with equal class numbers
            split = int(args.split * min_trials)
            x_val = dataset[:, :split, :, :]
            x_train = dataset[:, split:, :, :]

            x_train, mean, var = self.normalize(x_train)
            x_val, _, _ = self.normalize(x_val, mean, var)

            x_trains.append(x_train)
            x_vals.append(x_val)
            disconts.append([0])

        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, disconts

    def create_examples(self, x, disconts):
        '''
        Create examples with labels.
        '''
        x = x.transpose(1, 0)
        x = x.reshape(118, -1, self.timesteps, x.shape[1])
        x = x.transpose(0, 1, 3, 2)

        resample = int(1000/self.args.sr_data)
        x = x[:, :, :, ::resample]
        timesteps = x.shape[3]
        trials = x.shape[1]

        array = []
        labels = np.ones((trials, 1, timesteps))
        for c in range(x.shape[0]):
            array.append(np.concatenate((x[c, :, :, :], labels * c), axis=1))

        x = np.array(array).reshape(-1, x.shape[2] + 1, timesteps)
        return x


class CichyDataAttention(CichyData):
    def __init__(self, args):
        super(CichyDataAttention, self).__init__(args)
        self.x_train_t = self.x_train_t.permute(2, 0, 1)
        self.x_val = self.x_val_t.permute(2, 0, 1)
