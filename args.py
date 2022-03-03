import os
import torch
import torch.nn.functional as F
import numpy as np

from classifiers_simpleNN import SimpleClassifier
from classifiers_wavenet import WavenetClassifier
from classifiers_linear import LDA
from cichy_data import CichyData


class Args:
    gpu = '1'
    func = {'repeat_baseline': False,
            'AR_baseline': False,
            'LDA_baseline': False,
            'LDA_pairwise': False,
            'train': True,
            'generate': False,
            'recursive': False,
            'analyse_kernels': False,
            'kernel_network_FIR': False,
            'kernel_network_IIR': False,
            'plot_kernels': False,
            'feature_importance': False,
            'save_validation_ch': False,
            'save_validation_subs': False,
            'pca_sensor_loss': False,
            'PFIts': False,
            'PFIch': False,
            'compare_layers': False,
            'test': False,
            'LDA_eval': False,
            'PFIemb': False,
            'window_eval': False}

    def __init__(self):
        n = 20

        # training arguments
        self.name = 'args.py'
        self.load_dataset = True
        self.learning_rate = 0.00005
        self.max_trials = 1
        self.batch_size = 10
        self.epochs = 500
        self.val_freq = 20
        self.print_freq = 5
        self.num_plot = 1
        self.plot_ch = 1
        self.save_curves = True
        self.load_model = False
        self.result_dir = [os.path.join(
            'results',
            'disp_epoched',
            'subj2',
            'group_wavenetclass_cv20',
            'crossval' + str(i)) for i in range(n)]
        self.model = WavenetClassifier
        self.dataset = CichyData

        # wavenet arguments
        self.activation = torch.nn.Identity()
        self.linear = False
        self.subjects = 0
        self.embedding_dim = 0
        self.num_samples_CPC = 20
        self.p_drop = 0.7
        self.dropout2d_bad = False
        self.k_CPC = 1
        self.mu = 255
        self.ch_mult = 1
        self.groups = 1
        self.conv1x1_groups = 1
        self.kernel_size = 2
        self.timesteps = 1
        self.sample_rate = [100, 356]
        self.rf = 16
        rf = 16
        ks = self.kernel_size
        nl = int(np.log(rf) / np.log(ks))
        self.dilations = [ks**i for i in range(nl)]  # wavenet mode
        #self.dilations = [1] + [2] + [4] * 7  # costum dilations

        # classifier arguments
        self.load_conv = False
        self.l1_loss = False
        self.pred = False
        self.alpha_norm = 0.0
        self.num_classes = 5
        self.units = [500, 50]
        self.dim_red = 64
        self.stft_freq = 0

        # dataset arguments
        data_path = os.path.join('/', 'gpfs2', 'well', 'woolrich', 'projects',
                                 'disp_csaky', 's2', 'preproc125hz', 'group_sub')
        self.data_path = [data_path] * n
        self.num_channels = list(range(307))
        self.crop = 1
        self.whiten = False
        self.group_whiten = False
        self.split = self.split = [np.array([i/n, (i+1)/n]) for i in range(n)]
        self.sr_data = 250
        self.num_components = 0
        self.resample = 7
        self.save_norm = True
        self.norm_path = os.path.join(data_path, 'norm_coeff')
        self.pca_path = os.path.join(data_path, 'pca128_model')
        self.load_pca = False
        self.dump_data = [os.path.join(data_path, 'train_data_cv' + str(i), 'c') for i in range(n)]
        self.load_data = self.dump_data

        # analysis arguments
        self.closest_chs = 20
        self.PFI_inverse = False
        self.pfich_timesteps = [0, 256]
        self.PFI_perms = 20
        self.halfwin = 15
        self.compare_model = False
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
        self.AR_load_path = os.path.join(
            'results',
            'mrc',
            '60subjects_notch_sensors_multiAR64')
