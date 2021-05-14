import os
import numpy as np
from mat4py import loadmat
import torch
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from cichy_data import CichyData


class DondersData(CichyData):
    def __init__(self, args):
        self.args = args
        self.shift = int(
            (args.sample_rate - args.timesteps) / args.kernel_size + 1)

        if args.load_data:
            data = pickle.load(open(args.load_data, 'rb'))
            self.x_train = data[0][args.num_channels, :]
            self.x_val = data[1][args.num_channels, :]
            self.x_train_t = data[2][:, args.num_channels, :].cuda()
            self.x_val_t = data[3][:, args.num_channels, :].cuda()

            args.num_channels = len(args.num_channels)
            self.set_common()

            return

        if args.load_pca:
            pca_model = pickle.load(open(args.pca_path, 'rb'))

        if 'sub' in args.data_path:
            paths = [args.data_path]
        else:
            paths = os.listdir(args.data_path)
            paths = [os.path.join(args.data_path, p) for p in paths]
            paths = [p for p in paths if os.path.isdir(p)]

        x_trains, x_vals, disconts = self.load_data(args, paths)

        self.x_train = np.concatenate(tuple(x_trains), axis=1)
        self.x_val = np.concatenate(tuple(x_vals), axis=1)

        if args.num_components and not args.load_pca:
            pca_model = PCA(args.num_components, random_state=69)
            pca_model.fit(self.x_train.transpose())
            pickle.dump(pca_model, open(args.pca_path, 'wb'))
            args.num_channels = args.num_components

        if args.num_components or args.load_pca:
            self.x_train = pca_model.transform(
                self.x_train.transpose()).transpose()
            x_val = pca_model.transform(
                self.x_val.transpose()).transpose()
            x_rec = pca_model.inverse_transform(x_val.transpose())
            plt.plot(self.x_val[0, :2000])
            plt.plot(x_rec[:2000, 0])
            plt.savefig(os.path.join(args.result_dir, 'pca.svg'),
                        format='svg', dpi=1200)
            plt.close('all')
            self.x_val = x_val

            self.x_train, self.mean, self.var = self.normalize(self.x_train)
            self.x_val, _, _ = self.normalize(self.x_val, self.mean, self.var)

        train_eps = []
        val_eps = []
        for x_train, x_val, discont in zip(x_trains, x_vals, disconts):
            if args.num_components or args.load_pca:
                x_train = pca_model.transform(x_train.transpose()).transpose()
                x_val = pca_model.transform(x_val.transpose()).transpose()
                x_train, _, _ = self.normalize(x_train, self.mean, self.var)
                x_val, _, _ = self.normalize(x_val, self.mean, self.var)

            val_ln = len(x_val[0])
            val_disconts = [i for i in discont if i < val_ln]
            val_eps.append(self.create_examples(x_val, val_disconts))

            train_disconts = [0] + [i - val_ln for i in discont if i >= val_ln]
            train_eps.append(self.create_examples(x_train, train_disconts))

        train_ep = np.concatenate(tuple(train_eps))
        val_ep = np.concatenate(tuple(val_eps))
        np.random.shuffle(train_ep)
        np.random.shuffle(val_ep)
        print('Good samples: ', sum([x.shape[1] for x in x_trains + x_vals]))
        print('Extracted samples: ',
              (train_ep.shape[0] + val_ep.shape[0]) * self.shift)

        self.x_train_t = torch.Tensor(train_ep).float().cuda()
        self.x_val_t = torch.Tensor(val_ep).float().cuda()

        dump = [self.x_train, self.x_val, self.x_train_t, self.x_val_t]
        pickle.dump(dump, open(os.path.join(args.data_path, 'data'), 'wb'))

        self.set_common()

    def load_data(self, args, paths):
        x_trains = []
        x_vals = []
        disconts = []
        for path in paths:
            print(path)
            mask_path = os.path.join(path, 'good_samples_new.mat')
            mask = np.array(loadmat(mask_path)['X'])

            d = []
            for i in range(len(mask)):
                if mask[i] == 1 and mask[i-1] == 0:
                    d.append(i - sum(abs(mask[:i] - 1)))
            disconts.append(d)

            data_path = os.path.join(path, 'preprocessed_data_new.mat')
            x_train = np.array(loadmat(data_path)['X'])[args.num_channels, :]
            x_train = x_train[:, mask.nonzero()[0]]

            x_val = x_train[:, :int(args.split * x_train.shape[1])]
            x_train = x_train[:, int(args.split * x_train.shape[1]):]

            x_train, mean, var = self.normalize(x_train)
            x_val, _, _ = self.normalize(x_val, mean, var)

            x_trains.append(x_train)
            x_vals.append(x_val)

        args.num_channels = len(args.num_channels)
        return x_trains, x_vals, disconts

    def create_examples(self, x, disconts):
        # each element is a continuous data segment
        x_segments = []
        if len(disconts) > 1:
            for i, m in enumerate(disconts[:-1]):
                if len(x[0, m:disconts[i + 1]]) > self.args.sample_rate:
                    x_segments.append(x[:, m:disconts[i + 1]])
            x_segments.append(x[:, disconts[-1]:])
        else:
            x_segments.append(x)

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
        np.random.shuffle(x_epochs)

        return x_epochs
