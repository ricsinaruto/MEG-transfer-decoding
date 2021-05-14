import numpy as np
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
import pickle
import torch


class CichyData:
    def __init__(self, args):
        self.args = args

        trials = 30
        x, y, num_examples, trial_list = self.load_data()

        print(x.shape)
        x = x[:, :args.num_channels, :]
        x = self._normalize(x, args.num_channels)

        # print('Max value', np.max(np.abs(x_train)))

        y = [y for _ in range(args.sample_rate)]
        y = np.array(y).transpose()

        low = int(trials * args.split)
        high = int(2*trials * args.split)
        cl = list(range(args.num_classes))

        x_val = [x[low+trials*i:high+trials*i] for i in cl]
        x_train = [x[:low]] + [x[high+trials*i:low+trials*(i+1)] for i in cl]

        x_val = np.concatenate(*x_val)
        x_train = np.concatenate(*x_train)
        perm_val = np.random.permutation(len(self.x_val))
        perm_train = np.random.permutation(len(self.x_train))

        self.x_val_t = torch.Tensor(x_val[perm_val]).float().cuda()
        self.x_train_t = torch.Tensor(x_train[perm_train]).float().cuda()
        self.x_val = x_val.transpose(1, 0, 2).reshape(args.num_channels, -1)
        self.x_train = x_train.transpose(1, 0, 2).reshape(args.num_channels, -1)

        y_val = [y[low+trials*i:high+trials*i] for i in cl]
        y_train = [y[:low]] + [y[high+trials*i:low+trials*(i+1)] for i in cl]
        self.y_val = torch.tensor(np.concatenate(*y_val)[perm_val],
                                  dtype=torch.long).cuda()
        self.y_train = torch.tensor(np.concatenate(*y_train)[perm_train],
                                    dtype=torch.long).cuda()

        self.set_common()

    def set_common(self):
        self.bs = self.args.batch_size
        self.train_batches = int(self.x_train_t.shape[0] / self.bs + 1)
        self.val_batches = int(self.x_val_t.shape[0] / self.bs + 1)

    def normalize(self, x, mean=None, var=None):
        x = x.transpose()
        mean = np.mean(x, axis=0) if mean is None else mean
        var = np.std(x, axis=0) if var is None else var
        x = (x - mean)/var
        return x.transpose(), mean, var

    def _normalize(self, x, channels):
        x = x.transpose(1, 0, 2).reshape(channels, -1)
        x = self.normalize(x)
        x = x.reshape(channels, -1, self.args.sample_rate).transpose(1, 0, 2)

        return x

    def get_batch(self, i, data):
        end = data.shape[0] if (i+1)*self.bs > data.shape[0] else (i+1)*self.bs
        return data[i*self.bs:end, :, :]

    def get_train_batch(self, i):
        return self.get_batch(i, self.x_train_t)

    def get_val_batch(self, i):
        return self.get_batch(i, self.x_val_t)

    def load_data(self,
                  permute=False,
                  trials=30,
                  tmin=0):

        tmax = self.args.resample + self.args.sample_rate
        if not self.args.load_pca:
            data = pickle.load(open(self.args.data_path, 'rb'))
            data = data[:self.args.num_classes*trials, :, tmin:tmax]
            data = data[:, :, ::self.args.resample]

            channels = data.shape[1]
        else:
            data = pickle.load(open(self.args.data_path, 'rb'))

        # sanity check
        '''
        for i in range(10):
            plt.plot(data[0, i, :])
        plt.savefig(
            os.path.join('results', 'channels.svg'), format='svg', dpi=1200)
        plt.close('all')
        '''

        if self.args.num_components and not self.args.load_pca:
            # normalize
            data = self._normalize(data, channels)

            pca_model = PCA(n_components=self.args.num_components,
                            random_state=69)
            pca = UnsupervisedSpatialFilter(pca_model, average=False)
            data = pca.fit_transform(data)

            path = self.args.data_path + '_pca' + str(self.args.num_components)
            pickle.dump(data, open(path, 'wb'))

        ones = [np.ones(trials, dtype=np.int32)*i
                for i in range(self.args.num_classes)]
        y_train = np.concatenate((ones))

        return data, y_train


class CichyDataAttention(CichyData):
    def __init__(self, args):
        super(CichyDataAttention, self).__init__(args)
        self.x_train_t = self.x_train_t.permute(2, 0, 1)
        self.x_val = self.x_val_t.permute(2, 0, 1)
