import torch
import numpy as np
import sys
import pywt

from mne.time_frequency import tfr_array_morlet

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from pyriemann.estimation import XdawnCovariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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


class XGBoost(LDA):
    def init_model(self):
        self.model = XGBClassifier(objective='multi:softmax',
                                   reg_lambda=25,
                                   reg_alpha=25)
        self.fit_pca = False


class XGBoostPramit(LDA):
    def init_model(self):
        self.fit_pca = False
        self.model = XGBClassifier(objective='multi:softmax',
                                   eta=0.5,
                                   max_depth=6,
                                   nthread=12,
                                   subsample=0.9,
                                   )


class XGBoost_hyperopt(XGBoost):
    def objective(self, space):
        clf = XGBClassifier(objective='multi:softmax',
                            reg_lambda=space['reg_lambda'],
                            eta=space['eta'],
                            n_estimators=space['n_estimators'],
                            max_depth=int(space['max_depth']),
                            gamma=space['gamma'],
                            reg_alpha=space['reg_alpha'],
                            min_child_weight=int(space['min_child_weight']),
                            colsample_bytree=space['colsample_bytree'])

        clf.fit(self.data[0][0], self.data[0][1],
                eval_set=self.data, eval_metric="auc",
                early_stopping_rounds=10, verbose=False)

        accuracy = clf.score(self.data[1][0], self.data[1][1])
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}

    def run(self, x_train, x_val, window=None):
        self.window = window
        x_train, x_val, y_train, y_val = self.transform_data(x_train, x_val)
        self.data = [(x_train, y_train), (x_val, y_val)]

        space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                 'gamma': hp.uniform('gamma', 1, 9),
                 'reg_alpha': hp.uniform('reg_alpha', 0, 10),
                 'reg_lambda': hp.uniform('reg_lambda', 0, 10),
                 'eta': hp.uniform('eta', 0.01, 0.2),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                 'n_estimators': 180}

        trials = Trials()

        best_hyperparams = fmin(fn=self.objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)

        space = best_hyperparams
        self.model = XGBClassifier(
            objective='multi:softmax',
            reg_lambda=space['reg_lambda'],
            eta=space['eta'],
            max_depth=int(space['max_depth']),
            gamma=space['gamma'],
            reg_alpha=space['reg_alpha'],
            min_child_weight=int(space['min_child_weight']),
            colsample_bytree=space['colsample_bytree'])
        self.model.fit(x_train, y_train)

        # validation accuracy
        acc = self.model.score(x_val, y_val)

        return acc, None, None


class LDA_average(LDA):
    def prep_lda(self, data):
        '''
        Reshape data for LDA.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])
        ave_len = self.args.ave_len
        num_ave = int(self.ts/ave_len)

        for i in range(num_ave):
            data[:, i, :] = np.mean(
                data[:, ave_len*i:ave_len*(i+1), :], axis=1)

        data = data[:, :num_ave, :]
        print(data.shape)
        data = data.reshape(data.shape[0], -1)

        return data


class LDA_cov(LDA):
    def prep_lda(self, data, metric=np.cov):
        '''
        Reshape data for LDA.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.transpose(0, 2, 1)

        data_cov = []
        for i in range(data.shape[0]):
            mat = np.triu(metric(data[i])).reshape(-1)
            data_cov.append(mat[mat != 0])

        return np.array(data_cov)


class LDA_corr(LDA_cov):
    def prep_lda(self, data, metric=np.cov):
        '''
        Reshape data for LDA.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.transpose(0, 2, 1)

        data_cov = []
        for i in range(data.shape[0]):
            mat = np.diag(metric(data[i])).reshape(-1)
            data_cov.append(mat)

        return np.array(data_cov)


class LDA_cov_normed(LDA_cov):
    def transform_data(self, x_train, x_val):
        '''
        Prepare and apply PCA and standardization if needed.
        '''
        self.lda_norm = False
        x_train, x_val, y_train, y_val = super(
            LDA_cov_normed, self).transform_data(x_train, x_val)

        if self.fit_pca:
            self.norm.fit(x_train)
        x_train = self.norm.transform(x_train)
        x_val = self.norm.transform(x_val)

        return x_train, x_val, y_train, y_val

    def eval(self, x_val, window=None):
        '''
        Evaluate an already trained LDA model.
        '''
        self.window = window
        x_val, y_val = self.prepare(x_val)

        if not self.args.load_conv:
            x_val = self.pca.transform(x_val)

        x_val = self.prep_lda(x_val)
        x_val = self.norm.transform(x_val)

        return self.model.score(x_val, y_val), x_val, y_val


class XGBoost_cov(LDA_cov, XGBoost):
    pass


class XGBoostPramit_cov(LDA_cov, XGBoostPramit):
    pass


class XGBoost_hyperopt_cov(LDA_cov, XGBoost_hyperopt):
    pass


class LDA_average_trials(LDA):
    def prep_lda(self, data):
        '''
        Reshape data for LDA.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])
        print(data.shape)
        data = data.reshape(data.shape[0], 4, -1, data.shape[2])
        data = np.mean(data, axis=1)

        if self.window is not None:
            data = data[:, self.window[0]:self.window[1], :]

        print(data.shape)
        data = data.reshape(data.shape[0], -1)

        return data


class LDA_cov_across_trials(LDA):
    def prep_lda(self, data):
        '''
        Reshape data for LDA.
        '''
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.reshape(data.shape[0], 4, -1, data.shape[2])
        #data = data.reshape(data.shape[0], 4, -1)

        data_cov = []
        for b in range(data.shape[0]):
            channels = []
            for c in range(data.shape[3]):
                mat = np.triu(np.cov(data[b, :, :, c])).reshape(-1)
                channels.append(mat[mat != 0])

            data_cov.append(np.array(channels))

        data_cov = np.array(data_cov)

        return data_cov.reshape(data.shape[0], -1)


class LDA_avg_trials_cov(LDA):
    def prep_lda(self, data):
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.reshape(data.shape[0], 4, -1, data.shape[2])
        data = np.mean(data, axis=1)

        data = data.transpose(0, 2, 1)

        data_cov = []
        for i in range(data.shape[0]):
            mat = np.triu(np.cov(data[i])).reshape(-1)
            data_cov.append(mat[mat != 0])

        return np.array(data_cov)


class LDA_riemann(LDA):
    def __init__(self, args):
        super(LDA_riemann, self).__init__(args)
        #self.cov = ERPCovariances(estimator='lwf')
        self.cov = XdawnCovariances(nfilter=2)
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

    def eval(self, x_val, window=None):
        '''
        Evaluate an already trained LDA model.
        '''
        self.window = window
        x_val, y_val = self.prepare(x_val)

        x_val = self.prep_lda(x_val)
        x_val = self.cov.transform(x_val)
        x_val = self.tangent.transform(x_val)

        return self.model.score(x_val, y_val), x_val, y_val


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


class LDA_wavelet_all(LDA_wavelet):
    def wavelet(self, data):
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.transpose(0, 2, 1)
        trials = data.shape[0]

        # STFT
        data = data.reshape(-1, data.shape[2])
        data = torch.fft.rfft(torch.Tensor(data))
        data = data.numpy()

        data = data[:, ::5]
        data = np.concatenate((data.real, data.imag), axis=1)

        return trials, data

    def prep_lda(self, data):
        '''
        Apply wavelet transform when preparing data.
        '''
        trials, data = self.wavelet(data)
        data = data.reshape(trials, -1, data.shape[1])
        print(data.shape)
        data = data.reshape(trials, -1)

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
        self.model = LinearSVC()

    def init_model(self):
        self.model = LinearSVC()
        self.fit_pca = False

    def predict(self, x_val, window=None, x_t=None):
        '''
        Predict class labels.
        '''
        _, x_val, y_val = self.eval(x_val, window)

        return self.model.decision_function(x_val), y_val


class linearSVM_wavelet_freq(LDA_wavelet_freq, linearSVM):
    '''
    Linear SVM model using the functionalities of the LDA class.
    '''
    pass


class SVM(LDA):
    '''
    SVM model using the functionalities of the LDA class.
    '''
    def __init__(self, args):
        super(SVM, self).__init__(args)
        self.model = SVC()


class SVM_db4(LDA):
    def init_model(self):
        self.model = SVC(kernel='poly', degree=2, C=100)
        self.fit_pca = False

    def wavelet(self, data):
        data = data.reshape(-1, self.ts, data.shape[1])
        data = data.transpose(0, 2, 1)
        trials = data.shape[0]

        # db4
        coeffs = pywt.wavedec(data, 'db4', level=7)
        coeffs[-1] = np.zeros_like(coeffs[-1])
        coeffs[-2] = np.zeros_like(coeffs[-2])

        #data_denoised = pywt.waverec(coeffs, 'db4')

        feats = []
        for c in coeffs[:-2]:
            rms = np.sqrt(np.mean(c**2, axis=-1))
            feats.append(rms)

        feats = np.array(feats).transpose(1, 0, 2)

        return trials, feats

    def transform_data(self, x_train, x_val):
        '''
        Prepare and apply PCA and standardization if needed.
        '''
        self.lda_norm = False
        x_train, x_val, y_train, y_val = super(
            SVM_db4, self).transform_data(x_train, x_val)

        if self.fit_pca:
            self.norm.fit(x_train)
        x_train = self.norm.transform(x_train)
        x_val = self.norm.transform(x_val)

        return x_train, x_val, y_train, y_val

    def prep_lda(self, data):
        '''
        Apply wavelet transform when preparing data.
        '''
        trials, data = self.wavelet(data)
        #print(data.shape)
        data = data.reshape(trials, -1)

        print('Data shape: ', data.shape)

        return data

    def eval(self, x_val, window=None):
        '''
        Evaluate an already trained LDA model.
        '''
        self.window = window
        x_val, y_val = self.prepare(x_val)

        if not self.args.load_conv:
            x_val = self.pca.transform(x_val)

        x_val = self.prep_lda(x_val)
        x_val = self.norm.transform(x_val)

        return self.model.score(x_val, y_val), x_val, y_val


class LDA_db4(SVM_db4):
    def init_model(self):
        self.model = LinearDiscriminantAnalysis(solver='lsqr',
                                                shrinkage='auto')
        self.fit_pca = False


class QDA(LDA):
    '''
    QDA model using the functionalities of the LDA class.
    '''
    def __init__(self, args):
        super(QDA, self).__init__(args)
        self.model = QuadraticDiscriminantAnalysis()
