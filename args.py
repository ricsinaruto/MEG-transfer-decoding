import os
import torch
import numpy as np

from wavenets_simple import Conv1PoolNet, ConvPoolNet
from wavenets_full import WavenetFull, WavenetFullSimple
from simulated_data import EventSimulation, EventSimulationQuantized


class Args:
    gpu = '1'
    func = {'repeat_baseline': False,
            'AR_baseline': False,
            'train': True,
            'generate': False,
            'recursive': False,
            'analyse_kernels': False,
            'kernel_network_FIR': False,
            'kernel_network_IIR': False,
            'plot_kernels': False}

    def __init__(self):
        # training arguments
        self.learning_rate = 0.001
        self.batch_size = 10000
        self.epochs = 5000
        self.split = 0.2
        self.val_freq = 50
        self.print_freq = 1
        self.num_plot = 1
        self.plot_ch = 1
        self.save_curves = True
        self.load_model = False
        self.result_dir = os.path.join(
            'results',
            'simulated',
            '8event_snr1_convpool')
        self.model = ConvPoolNet
        self.dataset = EventSimulation

        # wavenet arguments
        self.activation = torch.asinh
        self.linear = False
        self.num_samples_CPC = 20
        self.p_drop = 0
        self.k_CPC = 1
        self.mu = 255
        self.ch_mult = 4
        self.groups = 1
        self.kernel_size = 16
        self.timesteps = 1
        self.num_classes = 118
        self.sample_rate = 497 + self.timesteps
        self.rf = 497
        ks = self.kernel_size
        nl = int(np.log(self.rf) / np.log(ks))
        self.dilations = [ks**i for i in range(nl)]  # wavenet mode
        self.dilations = [1] * 5  # no dilations

        # dataset arguments
        self.data_path = os.path.join('donders', '')
        self.num_channels = list(range(1))
        self.crop = 1
        self.sr_data = 250
        self.num_components = 128
        self.resample = 7
        self.pca_path = os.path.join(self.data_path, 'pca_model')
        self.load_pca = False
        self.load_data = os.path.join(
            'data', 'simulated', '8event_snr1', 'data.mat')

        # analysis arguments
        self.generate_noise = 0.53
        self.generate_length = self.sr_data * 1000
        self.generate_mode = 'IIR'
        self.generate_input = 'gaussian_noise'
        self.individual = True
        self.anal_lr = 0.05
        self.anal_epochs = 200
        self.norm_coeff = 0.002
        self.kernel_limit = 100

        # simulation arguments
        self.nonlinear_prenoise = True
        self.nonlinear_data = True
        self.seconds = 15000
        self.events = 12
        self.sim_num_channels = 1
        self.sim_ar_order = 2
        self.gamma_shape = 10
        self.gamma_scale = 10
        self.noise_std = 2.5
        self.lambda_exp = 0.005
        self.ar_shrink = 1.0
        self.freqs = [10, 14, 18, 22, 26, 33, 38, 45]
        self.ar_noise_std = np.random.rand(self.events) / 5 + 0.8
        self.max_len = 500

        # AR model arguments
        self.order = 64
        self.uni = True
        self.ar_plot_ts = list(range(1000))
