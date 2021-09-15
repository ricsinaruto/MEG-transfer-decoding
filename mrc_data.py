import os
import copy
import torch
import mat73
import numpy as np
from scipy.io import loadmat

from donders_data import DondersData


class MRCData(DondersData):
    '''
    Class for loading and processing resting state data from the MRC dataset.
    '''
    def load_mat_data(self, args):
        '''
        Loads ready-to-train splits from mat files.
        '''
        chn = args.num_channels
        x_trains = []
        x_vals = []
        x_train_ts = []
        x_val_ts = []

        # load data for each channel
        for index, i in enumerate(chn):
            data = loadmat(args.load_data + 'ch' + str(i) + '.mat')
            x_trains.append(np.array(data['x_train']))
            x_vals.append(np.array(data['x_val']))
            x_train_ts.append(np.array(data['x_train_t']))
            x_val_ts.append(np.array(data['x_val_t']))

            if index == 0:
                self.sub_id['train'] = np.array(data['sub_id_train'])
                self.sub_id['val'] = np.array(data['sub_id_val'])

        self.x_train = np.concatenate(tuple(x_trains))
        self.x_val = np.concatenate(tuple(x_vals))
        self.x_train_t = np.concatenate(tuple(x_train_ts), axis=1)
        self.x_val_t = np.concatenate(tuple(x_val_ts), axis=1)

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
        x_trains = []
        x_vals = []
        disconts = []
        for path in paths:
            print(path)
            if args.numpy:
                x_train = np.load(path).T
                d = np.array([0])
            else:
                try:
                    dat = loadmat(path)
                except NotImplementedError:
                    dat = mat73.loadmat(path)

                # discontinuous segment lengths are saved in T
                T = np.array(dat['T'])
                try:
                    _ = T.shape[1]
                    d = T[0].astype(int)
                except:
                    d = T.astype(int)

                try:
                    _ = len(d)
                except TypeError as e:
                    d = np.array([0])
                    print(str(e))

                x_train = np.transpose(np.array(dat['X']))

            for i, val in enumerate(d[1:]):
                d[i+1] = d[i] + val
            disconts.append(list((d/resample).astype(int)))

            # choose first 306 channels and downsample
            x_train = x_train[args.num_channels, ::resample]

            # create training and validation splits
            x_val = x_train[:, :int(args.split * x_train.shape[1])]
            x_train = x_train[:, int(args.split * x_train.shape[1]):]

            x_train, mean, var = self.normalize(x_train)
            x_val, _, _ = self.normalize(x_val, mean, var)

            x_trains.append(x_train)
            x_vals.append(x_val)

        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, disconts
