import numpy as np
from mat4py import loadmat

from donders_data import DondersData


class MRCData(DondersData):
    def load_data(self, args, paths):
        '''
        Load raw data from multiple subjects (paths).
        '''
        resample = int(1000/args.sr_data)
        x_trains = []
        x_vals = []
        disconts = []
        for path in paths:
            print(path)
            dat = loadmat(path)

            # discontinuous segment lengths are saved in T
            d = np.array(dat['T'])
            for i, val in enumerate(d):
                d[i] = int(sum(d[:i+1])/resample)
            disconts.append(d)

            # choose first 306 channels and downsample
            x_train = np.transpose(np.array(dat['X']))
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
