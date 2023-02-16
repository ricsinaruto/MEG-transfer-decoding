import os
import torch
import numpy as np
import pickle

from scipy.io import savemat
from mat4py import loadmat

from donders_data import DondersData


class EventSimulation(DondersData):
    '''
    Class for generating event-based simulated data where each event
    has a different frequency generated by a damped AR2 model from either
    deterministic or stochastic random noise. Lifetimes are sampled from
    a gamma distribution and the switching between events occurs
    according to a random or manually specified transition probability matrix.
    '''
    def __init__(self, args):
        '''
        Either create simulated data or load already created data.
        '''
        self.args = args

        if args.load_data:
            self.load_data(args)
            self.set_common()
            return

        # gamma contains the event lifetimes
        self.gamma = np.random.gamma(args.gamma_shape,
                                     args.gamma_scale,
                                     args.seconds)
        self.gamma = self.gamma.astype(int)

        # limit the longest lifetime to max_len
        self.gamma = np.array([g for g in self.gamma if g < args.max_len])

        # create the data and the gaussian noise arrays
        data_length = np.sum(self.gamma) + args.sim_ar_order
        self.data = np.random.randn(args.sim_num_channels, data_length)
        self.noise = np.random.normal(
            0, args.noise_std, (args.sim_num_channels, data_length))
        self.ar_noise_std = np.array(args.ar_noise_std)

        self.fill(args)
        self.generate(args)

        # run_fft(args, self.data[0, :], 'unfiltered_input_freq.svg')

        # save the simulated data and the state time course
        savemat(os.path.join(args.result_dir, 'data.mat'), {'X': self.data})
        path = os.path.join(args.result_dir, 'stc')
        pickle.dump(self.stc, open(path, 'wb'))

    def fill(self, args):
        '''
        Initialize the parameters of the simulation.
        '''
        # this makes sure that the AR2 model generates the required frequencies
        freqs = [2*np.cos(2*np.pi*f/args.sr_data) for f in args.freqs]
        self.AR = np.array([[f, -1] for f in freqs])
        self.AR *= args.ar_shrink
        self.AR = self.AR.reshape(args.events, 1, 1, args.sim_ar_order)

        probs = []
        self.deterministic = []
        for i in range(args.events):
            # initialize event transition probabilities for each event
            distribution = np.random.rand(args.events)
            probs.append(distribution/distribution.sum())

            # randomly select whether an event has
            # stochastic or deterministic AR noise
            if np.random.randint(0, 2):
                self.deterministic.append([])
            else:
                shape = (args.sim_num_channels, args.max_len)
                noise = np.random.normal(0, self.ar_noise_std[i], shape)
                self.deterministic.append(noise)

        self.transition = np.array(probs)

        # some special manual transition probabilities
        '''
        self.transition = np.array([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
                                    [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])
        self.transition = np.array([[0.01, 0.99, 0.0, 0.0],
                                    [0.0, 0.01, 0.99, 0.0],
                                    [0.0, 0.0, 0.01, 0.99],
                                    [0.99, 0.0, 0.0, 0.01]])
        '''

    def generate(self, args):
        '''
        Generate the simulated data using the predefined AR2 models.
        '''
        state = 0
        count = args.sim_ar_order
        self.stc = []

        # each element in gamma is the length of an event
        for lifetime in self.gamma:
            # whether to use stochastic or deterministic AR noise for the event
            if len(self.deterministic[state]):
                self.data[:, count:count+lifetime] = \
                    self.deterministic[state][:, :lifetime]
            else:
                shape = (args.sim_num_channels, lifetime)
                noise = np.random.normal(0, self.ar_noise_std[state], shape)
                self.data[:, count:count+lifetime] = noise

            # generate with AR2 recursively
            for t in range(lifetime):
                past = self.data[:, count+t-args.sim_ar_order:count+t]
                coeff = self.AR[state, :, :, :]
                self.data[:, count+t] += np.einsum(
                    'iij,ij->i', coeff, past[:, ::-1])

            # apply exponential damping to the current event
            zed = self.data[:, count:count+lifetime]
            time = np.array(range(1, lifetime+1)).reshape(1, -1)
            time = np.repeat(time, args.sim_num_channels, axis=0)
            lambda_ = args.lambda_exp
            self.data[:, count:count+lifetime] = np.exp(-lambda_ * time) * zed

            count += lifetime
            self.stc.extend([state]*lifetime)

            # sample next state
            state = np.argmax(np.random.multinomial(1, self.transition[state]))

        # normalize states separately
        '''
        stc = np.array(self.stc)
        for i in range(args.events):
            self.data[:, np.where(stc == i)], _, _ = \
                self.normalize(self.data[:, np.where(stc == i)])
        '''

        # apply nonlinearity to simulated data
        if args.nonlinear_data and args.nonlinear_prenoise:
            self.data = np.arcsinh(self.data)

        # finally add the gaussian noise
        self.data = self.data + self.noise

    def load_data(self, args):
        '''
        Load the already created simulate data.
        '''
        # shift is the input length minus the receptive field
        self.shift = args.sample_rate - args.timesteps - args.rf + 1
        args.num_channels = len(args.num_channels)

        # load data and apply nonlinearity if needed
        x_train = np.array(loadmat(args.load_data)['X'])
        x_train = x_train.reshape(args.num_channels, -1)
        if args.nonlinear_data and not args.nonlinear_prenoise:
            x_train = np.arcsinh(x_train)

        # if crop is less than 1, only use a portion of the data
        x_train = x_train[:, :int(args.crop*x_train.shape[1])]

        # create train and validation sets and normalize them
        x_val = x_train[:, :int(args.split * x_train.shape[1])]
        x_train = x_train[:, int(args.split * x_train.shape[1]):]
        self.x_train, mean, var = self.normalize(x_train)
        self.x_val, _, _ = self.normalize(x_val, mean, var)

        full_data = (self.x_train, self.x_val)
        self.maxval = np.amax(np.concatenate(full_data, axis=1)) + 0.1

        # create train and validation examples
        train_ep = self.create_examples(self.x_train, [])
        val_ep = self.create_examples(self.x_val, [])
        self.x_train_t = torch.Tensor(train_ep).float().cuda()
        self.x_val_t = torch.Tensor(val_ep).float().cuda()

        # load the generated state time course
        path = os.path.join(args.result_dir, 'stc')
        if os.path.isfile(path):
            self.stc = np.array(pickle.load(open(path, 'rb')))


class EventSimulationFixLifetimes(EventSimulation):
    def __init__(self, args):
        '''
        Either create simulated data or load already created data.
        '''
        if args.load_data:
            self.load_data(args)
            self.set_common()
            return

        # limit the longest lifetime to max_len
        self.gamma = 225 + 25 * np.random.rand(args.seconds)
        self.gamma = self.gamma.astype(int)

        # create the data and the gaussian noise arrays
        data_length = np.sum(self.gamma) + args.sim_ar_order
        self.data = np.random.randn(args.sim_num_channels, data_length)
        self.noise = np.random.normal(
            0, args.noise_std, (args.sim_num_channels, data_length))
        self.ar_noise_std = np.array(args.ar_noise_std)

        self.fill(args)
        self.generate(args)

        # run_fft(args, self.data[0, :], 'unfiltered_input_freq.svg')

        # save the simulated data and the state time course
        savemat(os.path.join(args.result_dir, 'data.mat'), {'X': self.data})
        path = os.path.join(args.result_dir, 'stc')
        pickle.dump(self.stc, open(path, 'wb'))


class EventSimulationQuantized(EventSimulation):
    '''
    This class handles the event-based simulated data in the quantized domain.
    Only works together with the quantized wavenet model.
    '''
    def load_data(self, args):
        '''
        Load data created by the EventSimulation class.
        '''
        super(EventSimulationQuantized, self).load_data(args)

        args.num_channels = args.mu + 1

        # save the unquantized data to different variables
        self.x_train_o = self.x_train_t.cpu().numpy()
        self.x_val_o = self.x_val_t.cpu().numpy()

        # quantized and encode the raw data
        self.x_train_t = self.one_hot_encode(self.quantize(self.x_train_o))
        self.x_val_t = self.one_hot_encode(self.quantize(self.x_val_o))
        self.x_train_t = torch.Tensor(self.x_train_t).float().cuda()
        self.x_val_t = torch.Tensor(self.x_val_t).float().cuda()

    def quantize(self, x):
        '''
        Quantize x using the mu-law algorithm.
        '''
        mu = self.args.mu
        x = x / self.maxval
        x = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)

        bins = np.linspace(-1, 1, mu + 1)
        x = np.digitize(x, bins) - 1

        return x

    def dequantize(self, x):
        '''
        Apply the inverse of the mu-law algorithm to get back the signal.
        '''
        mu = self.args.mu
        x = x / (mu + 1) * 2 - 1
        x = np.sign(x) * (np.exp(np.abs(x) * np.log(mu+1))-1) / mu

        return x * self.maxval

    def one_hot_encode(self, x):
        '''
        Apply one-hot encoding to the quantized data.
        '''
        one_hot = np.zeros((x.shape[0], self.args.mu + 1, x.shape[2]))

        for i in range(x.shape[0]):
            one_hot[i, x[i].ravel(), np.arange(x.shape[2])] = 1

        return one_hot
