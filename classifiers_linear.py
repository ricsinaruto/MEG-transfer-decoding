import torch
import numpy as np
import sys

from mne.time_frequency import tfr_array_morlet

from pyriemann.estimation import XdawnCovariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class LDA:
    '''
    Class implementing an LDA model with extra functionalities.
    '''
    def __init__(self, args):
        self.args = args
        self.lda_norm = True
        self.init_model()
        self.fit_pca = True
        self.pca = PCA(args.dim_red)
        self.norm = StandardScaler()
        self.spatial_conv = torch.nn.Identity()

        # if needed load a model for dimensionality reduction instead of PCA
        if args.load_conv:
            try:
                model = torch.load(args.load_conv)
                model.loaded(args)
                model.cuda()
                self.spatial_conv = model.spatial_conv
            except Exception:
                print('Couldn\'t load conv model for lda.')

    def init_model(self):
        self.model = LinearDiscriminantAnalysis(solver='lsqr',
                                                shrinkage='auto')
        self.fit_pca = False

    def loaded(self, args):
        self.args = args
        self.lda_norm = True

    def run(self, x_train, x_val, window=None):
        '''
        Transform data, then train and evaluate LDA.
        '''
        self.window = window
        x_train, x_val, y_train, y_val = self.transform_data(x_train, x_val)

        # fit LDA
        self.model.fit(x_train, y_train)

        # validation accuracy
        acc = self.model.score(x_val, y_val)

        return acc, None, None

    def transform_data(self, x_train, x_val):
        '''
        Prepare and apply PCA and standardization if needed.
        '''
        # prepare data specifically for sklearn classifiers
        x_train, y_train = self.prepare(x_train)
        x_val, y_val = self.prepare(x_val)

        # apply PCA if not using convolutional layer
        if not self.args.load_conv:
            if self.fit_pca:
                self.pca.fit(x_train)
            x_train = self.pca.transform(x_train)
            x_val = self.pca.transform(x_val)

            # standardize the output of PCA
            if self.lda_norm:
                if self.fit_pca:
                    self.norm.fit(x_train)
                x_train = self.norm.transform(x_train)
                x_val = self.norm.transform(x_val)

        # reshape data for LDA
        x_train = self.prep_lda(x_train)
        x_val = self.prep_lda(x_val)

        return x_train, x_val, y_train, y_val

    def get_output(self, x_val, window=None):
        '''
        Get LDA-ready input.
        '''
        _, x_val, y_val = self.eval(x_val, window)

        #output = self.model.decision_function(x_val) - self.model.intercept_
        output = x_val

        return x_val, y_val, output

    def predict(self, x_val, window=None, x_t=None):
        '''
        Predict class labels.
        '''
        _, x_val, y_val = self.eval(x_val, window)

        return self.model.predict_proba(x_val), y_val

    def eval(self, x_val, window=None):
        '''
        Evaluate an already trained LDA model.
        '''
        self.window = window
        x_val, y_val = self.prepare(x_val)

        if not self.args.load_conv:
            x_val = self.pca.transform(x_val)
            if self.lda_norm:
                x_val = self.norm.transform(x_val)

        x_val = self.prep_lda(x_val)
        return self.model.score(x_val, y_val), x_val, y_val

    def prepare(self, data):
        '''
        Extract labels and reshape data.
        '''
        self.ts = data.shape[2]
        ch = self.args.num_channels
        y = data[:, -1, 0].cpu().numpy()

        # if needed apply convolutional dimensionality reduction
        data = self.spatial_conv(data[:, :ch, :]).detach().cpu().numpy()

        data = data.transpose(0, 2, 1)
        data = data.reshape(-1, data.shape[2])

        return data, y

    def prep_lda(self, data):
        '''
        Reshape data for LDA.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])

        if self.window is not None:
            data = data[:, self.window[0]:self.window[1], :]
        data = data.reshape(data.shape[0], -1)

        return data


class LDA_riemann(LDA):
    def __init__(self, args):
        super(LDA_riemann, self).__init__(args)
        #self.cov = ERPCovariances(estimator='lwf')
        self.cov = XdawnCovariances(nfilter=8)
        self.tangent = TangentSpace(metric="riemann")

    def prep_lda(self, data):
        '''
        Reshape data for LDA.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])

        if self.window is not None:
            data = data[:, self.window[0]:self.window[1], :]

        data = data.transpose(0, 2, 1)

        return data

    def transform_data(self, x_train, x_val):
        '''
        Prepare and apply PCA and standardization if needed.
        '''
        # prepare data specifically for sklearn classifiers
        x_train, y_train = self.prepare(x_train)
        x_val, y_val = self.prepare(x_val)

        # reshape data for LDA
        x_train = self.prep_lda(x_train)
        x_val = self.prep_lda(x_val)

        # riemann transform
        if self.fit_pca:
            self.cov.fit(x_train, y_train)
        x_train = self.cov.transform(x_train)
        x_val = self.cov.transform(x_val)

        # project to tangent space
        if self.fit_pca:
            self.tangent.fit(x_train)
        x_train = self.tangent.transform(x_train)
        x_val = self.tangent.transform(x_val)

        return x_train, x_val, y_train, y_val


class LDA_wavelet(LDA):
    '''
    LDA model trained on wavelet transformed data.
    '''
    def wavelet(self, data):
        hw = self.args.halfwin
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.transpose(0, 2, 1)
        trials = data.shape[0]

        # STFT
        data = data.reshape(-1, data.shape[2])

        window = torch.hamming_window(2*hw)
        data = torch.stft(torch.Tensor(data),
                          n_fft=2*hw,
                          hop_length=1,
                          window=window,
                          center=False,
                          return_complex=False)
        data = data.numpy()
        data = data.transpose(0, 2, 1, 3)

        return trials, data

    def prep_lda(self, data):
        '''
        Apply wavelet transform when preparing data.
        '''
        trials, data = self.wavelet(data)
        data = data.reshape(data.shape[0], data.shape[1], -1)
        data = data.reshape(trials, -1, data.shape[1], data.shape[2])

        # morlet wavelet
        '''
        freqs = np.logspace(*np.log10([2, 40]), num=5)
        n_cycles = freqs / 4.

        data = tfr_array_morlet(data,
                                sfreq=self.args.sr_data,
                                freqs=freqs,
                                n_cycles=n_cycles)
        '''
        # select small window
        data = data[:, :, self.window[0], :].reshape(trials, -1)

        #data = np.append(data.real, data.imag, axis=1)
        print('Data shape: ', data.shape)

        return data


class LDA_wavelet_freq(LDA_wavelet):
    def prep_lda(self, data):
        trials = data.shape[0]
        trials, data = self.wavelet(data)

        data = data[:, :, self.args.stft_freq, :]
        data = data.reshape(trials, -1, data.shape[1], data.shape[2])
        data = data[:, :, self.window[0], :].reshape(trials, -1)

        print('Data shape: ', data.shape)
        return data


class LDA_wavelet_forest(LDA_wavelet_freq):
    def prep_lda(self, data):
        trials = data.shape[0]
        trials, data = self.wavelet(data)

        self.freqs = data.shape[2]
        data = data.reshape(
            trials, -1, data.shape[1], self.freqs, data.shape[3])

        data = data[:, :, self.window[0], :, :]
        data = data.transpose(2, 0, 1, 3)
        data = data.reshape(self.freqs, trials, -1)

        print('Data shape: ', data.shape)
        return data

    def run(self, x_train, x_val, window=None):
        '''
        Transform data, then train and evaluate LDA.
        '''
        self.window = window
        x_train, x_val, y_train, y_val = self.transform_data(x_train, x_val)

        # fit LDA for each frequency band
        y_preds_t = []
        y_preds_v = []
        for f in range(self.freqs):
            self.init_model()
            self.model.fit(x_train[f, :, :], y_train)

            y_preds_t.append(self.model.predict(x_train[f, :, :]))
            y_preds_v.append(self.model.predict(x_val[f, :, :]))

        # fit random forest on all frequencies
        self.forest = RandomForestClassifier()
        y_preds_t = np.array(y_preds_t)
        y_preds_v = np.array(y_preds_v)

        self.forest.fit(y_preds_t.T, y_train)
        acc = self.forest.score(y_preds_v.T, y_val)

        return acc, y_preds_v.T, y_val

    def predict(self, x_val, window=None, x_train=None):
        '''
        Predict class labels.
        '''
        acc, y_preds, y_val = self.run(x_train, x_val, window)
        proba = self.forest.predict_proba(y_preds)

        return proba, y_val


class LogisticReg(LDA):
    '''
    Logistic Regression model using the functionalities of the LDA class.
    Uses L2 regularization.
    '''
    def init_model(self):
        self.model = LogisticRegression(multi_class='multinomial',
                                        max_iter=1000)
        self.fit_pca = False


class LogisticRegL1(LDA):
    '''
    Logistic Regression model using the functionalities of the LDA class.
    Uses L1 regularization.
    '''
    def __init__(self, args):
        super(LogisticRegL1, self).__init__(args)
        self.model = LogisticRegression(multi_class='ovr',
                                        penalty='l1',
                                        solver='liblinear')


class linearSVM(LDA):
    '''
    Linear SVM model using the functionalities of the LDA class.
    '''
    def __init__(self, args):
        super(linearSVM, self).__init__(args)
        self.model = LinearSVC(multi_class='crammer_singer', max_iter=2000)


class SVM(LDA):
    '''
    SVM model using the functionalities of the LDA class.
    '''
    def __init__(self, args):
        super(SVM, self).__init__(args)
        self.model = SVC()


class QDA(LDA):
    '''
    QDA model using the functionalities of the LDA class.
    '''
    def __init__(self, args):
        super(QDA, self).__init__(args)
        self.model = QuadraticDiscriminantAnalysis()
