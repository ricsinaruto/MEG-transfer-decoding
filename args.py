import os
import torch
import torch.nn.functional as F
import numpy as np

from classifiers_linear import LDA, LDA_average, LDA_cov, LDA_cov_sid, LDA_tp, LDA_sessnorm
from classifiers_simpleNN import SimpleClassifier
from wavenets_simple import WavenetSimple, ConvPoolNet
from classifiers_wavenet import WavenetClassifier, WavenetContClass, ConvPoolClassifier
from cichy_data import CichyData, CichyContData, CichySimpleContData, CichyDataNoNorm
from cichy_data import CichyDataCrossvalRobust, CichyDataRobust, CichyDataTrialNorm, CichyDataDISP


class Args:
    gpu = '0'  # cuda gpu index
    func = {'LDA_baseline': True}  # dict of functions to run from training.py

    def __init__(self):
        n = 1  # can be used to do multiple runs, e.g. over subjects

        # experiment arguments
        self.name = 'args.py'  # name of this file, don't change
        self.fix_seed = False
        self.common_dataset = False
        self.load_dataset = True  # whether to load self.dataset
        self.learning_rate = 0.00005  # learning rate for Adam
        self.anneal_lr = False  # whether to anneal learning rate
        self.max_trials = 1  # ratio of training data (1=max)
        self.batch_size = 20  # batch size for training and validation data
        self.epochs = 2000  # number of loops over training data
        self.val_freq = 20  # how often to validate (in epochs)
        self.print_freq = 5  # how often to print metrics (in epochs)
        self.save_curves = True  # whether to save loss curves to file
        self.load_model = False
        self.result_dir = [os.path.join(
            'results',
            'disp_epoched',
            'eeg',
            '1_40hz_noica_thinkall_4s',
            'lda_3sess')]
        self.model = LDA_sessnorm  # class of model to use
        self.dataset = CichyDataNoNorm  # dataset class for loading and handling data

        # wavenet arguments
        self.activation = torch.nn.Identity()  # activation function for models
        self.pooling = torch.nn.MaxPool1d
        self.subjects = 0  # number of subjects used for training
        self.embedding_dim = 0  # subject embedding size
        self.p_drop = 0.5  # dropout probability
        self.ch_mult = 2  # channel multiplier for hidden channels in wavenet
        self.kernel_size = 5  # convolutional kernel size
        self.timesteps = 1  # how many timesteps in the future to forecast
        self.sample_rate = [100, 200]  # start and end of timesteps within trials
        self.rf = 8  # receptive field of wavenet
        rf = 8
        ks = self.kernel_size
        nl = int(np.log(rf) / np.log(ks))
        self.dilations = [1] * 4  # dilation: 2^num_layers
        #self.dilations = [1] + [2] + [4] * 7  # costum dilations

        # classifier arguments
        self.wavenet_class = ConvPoolNet  # class of wavenet model
        self.load_conv = 'y' # where to load neural nerwork
        # dimensionality reduction from
        self.pred = False  # whether to use wavenet in prediction mode
        self.init_model = True  # whether to reinitialize classifier
        self.reg_semb = True  # whether to regularize subject embedding
        self.fixed_wavenet = False  # whether to fix weights of wavenet
        self.alpha_norm = 0.0  # regularization multiplier on weights
        self.num_classes = 5  # number of classes for classification
        self.units = [20, 10]  # hidden layer sizes of fully-connected block
        self.dim_red = 69  # number of pca components for channel reduction
        self.ave_len = 5
        self.stft_freq = 0  # STFT frequency index for LDA_wavelet_freq model
        self.decode_peak = 0.15
        self.decode_front = 0.45
        self.decode_back = 0.45
        self.epoch_ratio = 1
        self.label_smoothing = 0.0
        self.no_nonepoch = False

        # dataset arguments
        data_path = os.path.join('/', 'well', 'woolrich', 'projects',
                                 'disp_csaky', 'eeg',
                                 'preproc1_40hz_noica', 'thinkall_inner_speech2', 'goods')
        self.data_path = [os.path.join(data_path)]  # path(s) to data directory
        self.num_channels = list(range(53))#[36,31,5,167,174,28,63,177,33,19,30,140,47,45,166,130,121,168,145,131,132,200,275,176,304,272,17,279,133,149,281,165,181,46,120,15,172,249,198,32,37,6,35,151,303,147,34,27,182,150,173,38,179,129,270,274,250,199,178,170,171,16,29,280,152,141,143,305,251,146,144,65,148,7,24,175,169,273,134,271,25,138,64,26,18,3,142,122,139,4,8,180,20,306]
        self.numpy = True  # whether data is saved in numpy format
        self.shuffle = False
        self.crop = 1  # cropping ratio for trials
        self.whiten = False  # pca components used in whitening
        self.group_whiten = False  # whether to perform whitening at the GL
        self.split = [np.array([0.0, 0.2]),
                       np.array([0.2, 0.4]),
                       np.array([0.4, 0.6]),
                       np.array([0.6, 0.8]),
                       np.array([0.8, 1.0])] # validation split (start, end)
        self.sr_data = 100  # sampling rate used for downsampling
        self.original_sr = 1000  # original sampling rate of data
        self.save_data = False  # whether to save the created data
        self.val_max_trials = False
        self.save_whiten = False
        self.subjects_data = False  # list of subject inds to use in group data
        self.dump_data = [os.path.join(
            self.data_path[i], 'standard_scaler_sr1000cv') for i in range(n)]  # path(s) for dumping data
        self.load_data = ''#self.dump_data  # path(s) for loading data files

        # analysis arguments
        self.closest_chs = 'notebooks/eeg_closest1'  # channel neighbourhood size for spatial PFI
        self.kernelPFI = False
        self.chn_multi = 1  # channel multiplier for spatial PFI
        self.PFI_inverse = False  # invert which channels/timesteps to shuffle
        self.pfich_timesteps = [[0, 4000]]  # time window for spatiotemporal PFI
        self.PFI_perms = 50  # number of PFI permutations
        self.PFI_step = 10
        self.PFI_val = True  # whether to use validation set for PFI
        self.halfwin = 10  # half window size for temporal PFI
        self.halfwin_uneven = False  # whether to use even or uneven window
        self.generate_noise = 1  # noise used for wavenet generation
        self.generate_length = self.sr_data * 1000  # generated timeseries len
        self.generate_mode = 'IIR'  # IIR or FIR mode for wavenet generation
        self.generate_input = 'gaussian_noise'  # input type for generation
        self.individual = True  # whether to analyse individual kernels
        self.anal_lr = 0.001  # learning rate for input backpropagation
        self.anal_epochs = 200  # number of epochs for input backpropagation
        self.norm_coeff = 0.0001  # L2 of input for input backpropagation
        self.kernel_limit = 300  # max number of kernels to analyse

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

        # unused
        self.num_plot = 1
        self.plot_ch = 1
        self.linear = False
        self.num_samples_CPC = 20
        self.dropout2d_bad = False
        self.k_CPC = 1
        self.mu = 255
        self.groups = 1
        self.conv1x1_groups = 1
        self.pos_enc_type = 'cat'
        self.pos_enc_d = 128
        self.l1_loss = False
        self.norm_alpha = self.alpha_norm
        self.num_components = 0
        self.resample = 7
        self.save_norm = True
        self.norm_path = os.path.join(data_path, 'norm_coeff')
        self.pca_path = os.path.join(data_path, 'pca128_model')
        self.load_pca = False
        self.compare_model = False
        self.channel_idx = 0
