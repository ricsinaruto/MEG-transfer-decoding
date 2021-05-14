import sys
import os
import numpy as np
import mne
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
import pickle
import matplotlib.pyplot as plt
import sails
from scipy.fft import fft, fftfreq
from scipy.signal import welch


def run_fft(args, data, fn, figsize=(40, 10)):
    sr = args.sr_data
    f, Pxx_den = welch(data, sr, nperseg=2*sr)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.semilogy(f, Pxx_den)
    plt.ylim([None, None])

    for freq in args.freqs:
        plt.axvline(x=freq, color='red')

    path = os.path.join(args.result_dir, fn)
    plt.savefig(path, format='svg', dpi=1200)
    plt.close('all')


def run_standard_fft(args, data, fn, figsize=(40, 10)):
    N = len(data)
    sr = args.sr_data
    yf = fft(data)
    xf = fftfreq(N, 1/sr)[1:N//2]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot(xf, 2.0/N * np.abs(yf[1:N//2]), linewidth=0.5)
    ax.set_xticks(np.arange(0, int(xf[-1]), 1))

    for freq in args.freqs:
        plt.axvline(x=freq, color='red')

    path = os.path.join(args.result_dir, fn)
    fig.savefig(path, format='svg', dpi=1200)
    plt.close('all')


def test_target(target):
    with open('target_evaluate.txt', 'w') as f:
        targ = target.detach().cpu().reshape(-1)
        for i in range(targ.shape[0]):
            f.write(str(targ[i]))
            f.write('\n')

        plt.plot(targ.numpy()[:1000], linewidth=0.5)
        plt.savefig('target_evaluate.svg', format='svg', dpi=1200)
        plt.close('all')


def run_softmax(data, args):
    # data = torch.Tensor(data).float().cuda()
    # data = softmax(data, dim=1).cpu().numpy()
    print(sum(data[0, :, 0]))

    data = np.mean(data, axis=0)

    for c in range(args.num_channels):
        plt.plot(data[c, :], linewidth=1.0)

    plt.savefig('results/pca_softmax.svg', format='svg', dpi=1200)
    plt.close('all')

    sys.exit()


'''
def generation(self):
        self.model.eval()
        generation = []
        seq = self.x_val[0, :, 1:].view(1, 1, -1)
        for i in range(self.args.sample_rate * 10):
            output = self.model(seq).detach()
            # seq = output
            output = output[:, :, -1]
            generation.append(output.cpu())
            seq = torch.cat((seq[:, :, 1:], output.view(1, 1, -1)), 2)

        generation = torch.stack(generation).view(-1).numpy()

        self.plot_signals(self.x_val[1:11, 0, :].view(-1).cpu().numpy(),
                          generation,
                          self.args.result_dir + 'wavenet_gen.svg')
'''
'''
def ar_cholesky():
    # need to check what this code is doing exactly
    if False:
            C = np.linalg.cholesky(model.resid_cov)
            #print(C)
            x_val = x_val.transpose(0, 2, 1).reshape(-1, channels)
            x_val = x_val.dot(C)
            x_val = x_val.reshape(-1, sample_rate, channels).transpose(0, 2, 1)
'''


def load_data(path, num_components=0, permute=False, conditions=118, trials=30, resample=1, tmin=0, tmax=0, remove_epochs=False, filtering=False, load_pca=False):
    if not load_pca:
        data = pickle.load(open(path,'rb'))
        data = data[:conditions*trials, :, tmin:tmax]
        data = data[:, :, ::resample]
        
        channels = data.shape[1]
    else:
        data = pickle.load(open(path + '_pca', 'rb'))

    examples = conditions*trials

    if filtering:
        data = data.transpose(1, 0, 2).reshape(channels, -1)
        info = mne.create_info(ch_names=channels, ch_types='mag', sfreq=1000/resample)

        data = mne.io.RawArray(data, info)
        data.filter(1, 15)

        data = data.get_data()
        data = data.reshape(channels, examples, -1).transpose(1, 0, 2)

    # sanity check
    '''
    for i in range(10):
        plt.plot(data[0, i, :])
    plt.savefig(os.path.join('results', 'channels.svg'), format='svg', dpi=1200)
    plt.close('all')
    '''
    trial_list = []
    if remove_epochs:
        good_inds = sails.utils.detect_artefacts(data, axis=0, ret_mode='good_inds')
        data = data[good_inds, :, :]
        examples = data.shape[0]

        for i in range(conditions):
            trial_list.append(sum(good_inds[i*trials:(i+1)*trials]))

        #print(trial_list)

    if num_components and not load_pca:
        # normalize
        data = data.transpose(1, 0, 2).reshape(channels, -1).transpose(1, 0)
        data = (data - np.mean(data, axis=0))/np.std(data, axis=0)
        data = data.transpose(1, 0).reshape(channels, examples, -1).transpose(1, 0, 2)

        pca = UnsupervisedSpatialFilter(PCA(n_components=num_components, random_state=69), average=False)
        data = pca.fit_transform(data)
        pickle.dump(data, open(path + '_pca10', 'wb'))

    y_train = np.concatenate(([np.ones(trials, dtype=np.int32)*i for i in range(conditions)]))

    if permute:
        p = np.random.permutation(data.shape[0])
        data = data[p]
        y_train = y_train[p]

    return data, y_train, examples, trial_list
