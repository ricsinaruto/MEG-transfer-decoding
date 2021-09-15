import os
import torch
import torch.nn.functional as F
import numpy as np

from wavenets_simple import Conv1PoolNet, ConvPoolNet
from wavenets_simple import WavenetSimple, WavenetSimpleSembAdd, WavenetSimpleSembMult
from wavenets_classifier import WavenetClassifier, SimpleClassifier, WavenetClassPred
from wavenets_full import WavenetFull, WavenetFullSimple
from simulated_data import EventSimulation, EventSimulationQuantized
from mrc_data import MRCData
from cichy_data import CichyData


class Args:
    gpu = '0'
    func = {'repeat_baseline': False,
            'AR_baseline': False,
            'train': True,
            'generate': False,
            'recursive': False,
            'analyse_kernels': False,
            'kernel_network_FIR': False,
            'kernel_network_IIR': False,
            'plot_kernels': False,
            'feature_importance': False,
            'save_validation_ch': False,
            'save_validation_subs': False}

    def __init__(self):
        # training arguments
        self.name = 'args.py'
        self.learning_rate = 0.00005
        self.batch_size = 128
        self.epochs = 3000
        self.val_freq = 20
        self.print_freq = 5
        self.num_plot = 1
        self.plot_ch = 1
        self.save_curves = True
        self.load_model = [os.path.join(
            'results',
            'cichy_epoched',
            'subject1_mne_wavenet2ch_pretrainpred',)]
        self.result_dir = [os.path.join(
            'results',
            'cichy_epoched',
            'subject1_mne_wavenet2ch_pretrainpred',
            'classification')]
        self.model = WavenetClassPred
        self.dataset = CichyData

        # wavenet arguments
        self.activation = torch.asinh
        self.linear = False
        self.pred = False
        self.subjects = 0
        self.embedding_dim = 0
        self.num_samples_CPC = 20
        self.p_drop = 0.7
        self.k_CPC = 1
        self.mu = 255
        self.ch_mult = 2
        self.groups = 1
        self.conv1x1_groups = 1
        self.kernel_size = 2
        self.timesteps = 1
        self.num_classes = 118
        self.sample_rate = 257
        self.rf = 64
        rf = 512
        ks = self.kernel_size
        nl = int(np.log(self.rf) / np.log(ks))
        self.dilations = [ks**i for i in range(nl)]  # wavenet mode
        #self.dilations = [1] + [2] + [4] * 7  # costum dilations

        # dataset arguments
        data_path = os.path.join('/', 'gpfs2', 'well', 'woolrich', 'projects',
                                 'cichy118_cont', 'preproc_data_onepass', 'subj0')
        self.data_path = [os.path.join(data_path)]
        self.num_channels = list(range(307))
        self.crop = 1
        self.split = 0.2
        self.sr_data = 250
        self.num_components = 0
        self.resample = 7
        self.save_norm = True
        self.norm_path = [os.path.join(data_path, 'norm_coeff')]
        self.pca_path = [os.path.join(data_path, 'pca128_model')]
        self.load_pca = False
        self.dump_data = [os.path.join(data_path, 'train_data_trials', 'c')]
        self.load_data = self.dump_data

        # analysis arguments
        self.generate_noise = 1
        self.generate_length = self.sr_data * 1000
        self.generate_mode = 'IIR'
        self.generate_input = 'gaussian_noise'
        self.individual = True
        self.anal_lr = 0.001
        self.anal_epochs = 200
        self.norm_coeff = 0.0001
        self.kernel_limit = 300
        self.channel_idx = 0

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
        self.uni = False
        self.save_AR = False
        self.do_anal = False
        self.AR_load_path = [os.path.join(
            'results',
            'mrc',
            '60subjects_notch_sensors_multiAR64')]
