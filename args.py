import os
import torch
import torch.nn.functional as F
import numpy as np

from wavenets_simple import Conv1PoolNet, ConvPoolNet, WavenetSimple
from wavenets_full import WavenetFull, WavenetFullSimple
from simulated_data import EventSimulation, EventSimulationQuantized
from mrc_data import MRCData


class Args:
    gpu = '0'
    func = {'repeat_baseline': False,
            'AR_baseline': False,
            'train': False,
            'generate': False,
            'recursive': True,
            'analyse_kernels': False,
            'kernel_network_FIR': False,
            'kernel_network_IIR': False,
            'plot_kernels': False}

    def __init__(self):
        # training arguments
        self.learning_rate = 0.00005
        self.batch_size = 800
        self.epochs = 5000
        self.val_freq = 50
        self.print_freq = 1
        self.num_plot = 1
        self.plot_ch = 1
        self.save_curves = True
        self.load_model = True
        self.result_dir = os.path.join(
            'results',
            'mrc',
            'subject1_pca69_simplewave8ch69group16ks')
        self.model = WavenetSimple
        self.dataset = MRCData

        # wavenet arguments
        self.activation = torch.asinh
        self.linear = False
        self.num_samples_CPC = 20
        self.p_drop = 0.0
        self.k_CPC = 1
        self.mu = 255
        self.ch_mult = 8
        self.groups = 69
        self.kernel_size = 16
        self.timesteps = 1
        self.num_classes = 118
        self.sample_rate = 2*466 + self.timesteps
        self.rf = 466
        ks = self.kernel_size
        nl = int(np.log(self.rf) / np.log(ks))
        self.dilations = [ks**i for i in range(nl)]  # wavenet mode
        self.dilations = [1] + [2] + [4] * 7  # costum dilations

        # dataset arguments
        data_path = os.path.join('/', 'gpfs2', 'well', 'woolrich', 'projects',
                                 'mrc_meguk', 'eo-elekta',
                                 'preproc_nonotch.opt', 'mat_data')
        self.data_path = os.path.join(data_path, 'subject1.mat')
        self.num_channels = list(range(69))
        self.crop = 1
        self.split = 0.2
        self.sr_data = 250
        self.num_components = 69
        self.resample = 7
        self.pca_path = os.path.join(data_path, 'subject1', 'pca69_model_subject1')
        self.load_pca = True
        self.dump_data = os.path.join(data_path, 'subject1', 'sub1pca69rf466.mat')
        self.load_data = self.dump_data

        # analysis arguments
        self.generate_noise = 0.74
        self.generate_length = self.sr_data * 500
        self.generate_mode = 'IIR'
        self.generate_input = 'gaussian_noise'
        self.individual = True
        self.anal_lr = 0.05
        self.anal_epochs = 200
        self.norm_coeff = 0.002
        self.kernel_limit = 300
        self.channel_idx = 6

        # simulation arguments
        self.nonlinear_prenoise = True
        self.nonlinear_data = True
        self.seconds = 3000
        self.events = 8
        self.sim_num_channels = 1
        self.sim_ar_order = 2
        self.gamma_shape = 14
        self.gamma_scale = 14
        self.noise_std = 2.5
        self.lambda_exp = 0.005
        self.ar_shrink = 1.0
        self.freqs = []
        self.ar_noise_std = np.random.rand(self.events) / 5 + 0.8
        self.max_len = 1000

        # AR model arguments
        self.order = 64
        self.uni = True
