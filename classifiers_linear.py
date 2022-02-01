import torch
import numpy as np

from mne.time_frequency import tfr_array_morlet

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class LDA:
    '''
    Class implementing an LDA model with extra functionalities.
    '''
    def __init__(self, args):
        self.args = args
        self.init_model()
        self.fit_pca = True
        self.pca = PCA(args.dim_red)
        self.norm = StandardScaler()
        self.spatial_conv = torch.nn.Identity()

        # if needed load a model for dimensionality reduction instead of PCA
        if args.load_conv:
            model = torch.load(args.load_conv)
            model.loaded(args)
            model.cuda()
            self.spatial_conv = model.spatial_conv

    def init_model(self):
        self.model = LinearDiscriminantAnalysis(solver='lsqr',
                                                shrinkage='auto')
        self.fit_pca = False

    def crop(self, xt, xv):
        xt = xt[:, :, self.window[0]:self.self.window[1]]
        xv = xv[:, :, self.window[0]:self.self.window[1]]

        return xt, xv

    def run(self, x_train, x_val, window=None):
        '''
        Transform data, then train and evaluate LDA.
        '''
        self.window = window
        x_train, x_val = self.crop(x_train, x_val)
        x_train, x_val, y_train, y_val = self.transform_data(x_train, x_val)

        # fit LDA
        self.model.fit(x_train, y_train)

        # validation accuracy
        acc = self.model.score(x_val, y_val)

        return acc

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
            if self.fit_pca:
                self.norm.fit(x_train)
            x_train = self.norm.transform(x_train)
            x_val = self.norm.transform(x_val)

        # reshape data for LDA
        x_train = self.prep_lda(x_train)
        x_val = self.prep_lda(x_val)

        return x_train, x_val, y_train, y_val

    def eval(self, x_val):
        '''
        Evaluate an already trained LDA model.
        '''
        x_val, y_val = self.prepare(x_val)

        if not self.args.load_conv:
            x_val = self.pca.transform(x_val)
            x_val = self.norm.transform(x_val)

        x_val = self.prep_lda(x_val)
        return self.model.score(x_val, y_val)

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
        data = data.reshape(data.shape[0], -1)

        return data


class LDA_wavelet(LDA):
    '''
    LDA model trained on wavelet transformed data.
    '''
    def crop(self, xt, xv):
        return xt, xv

    def prep_lda(self, data):
        '''
        Apply wavelet transform when preparing data.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.transpose(0, 2, 1)

        # morlet wavelet
        freqs = np.logspace(*np.log10([2, 40]), num=5)
        n_cycles = freqs / 4.

        data = tfr_array_morlet(data,
                                sfreq=self.args.sr_data,
                                freqs=freqs,
                                n_cycles=n_cycles)
        # select small window
        data = data[:, :, :, self.window[0]:self.window[1]]

        data = data.reshape(data.shape[0], -1)
        data = np.append(data.real, data.imag, axis=1)
        print('Data shape: ', data.shape)

        return data


class LogisticReg(LDA):
    '''
    Logistic Regression model using the functionalities of the LDA class.
    Uses L2 regularization.
    '''
    def __init__(self, args):
        super(LogisticReg, self).__init__(args)
        self.model = LogisticRegression(multi_class='multinomial',
                                        max_iter=1000)


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
