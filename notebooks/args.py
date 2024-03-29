import os
import torch
import torch.nn.functional as F
import numpy as np

from classifiers_linear import LDA, LogisticReg, SVM, LDA_cov_featnorm


class Args:
    gpu = '1'  # cuda gpu index
    func = {'LDA_baseline': True}  # dict of functions to run from training.py

    def __init__(self):
        n = 1  # can be used to do multiple runs, e.g. over subjects

        # experiment arguments
        self.name = 'args.py'  # name of this file, don't change
        self.fix_seed = False
        self.common_dataset = False
        self.load_dataset = True  # whether to load self.dataset
        self.learning_rate = 0.0001  # learning rate for Adam
        self.max_trials = 1  # ratio of training data (1=max)
        self.val_max_trials = False
        self.batch_size = 20  # batch size for training and validation data
        self.epochs = 5000  # number of loops over training data
        self.val_freq = 20  # how often to validate (in epochs)
        self.print_freq = 5  # how often to print metrics (in epochs)
        self.save_curves = True  # whether to save loss curves to file
        self.load_model = False
        self.result_dir = os.path.join(
            '..',  # path(s) to save model and others
            'results',
            'stream_test')
        self.model = LDA_cov_featnorm  # class of model to use
        self.dataset = None  # dataset class for loading and handling data

        # wavenet arguments
        self.activation = torch.nn.Identity()  # activation function for models
        self.subjects = 1  # number of subjects used for training
        self.embedding_dim = 0  # subject embedding size
        self.p_drop = 0.6  # dropout probability
        self.ch_mult = 2  # channel multiplier for hidden channels in wavenet
        self.kernel_size = 2  # convolutional kernel size
        self.timesteps = 1  # how many timesteps in the future to forecast
        self.sample_rate = [0, 100]  # start and end of timesteps within trials
        self.rf = 256  # receptive field of wavenet
        rf = 128
        ks = self.kernel_size
        nl = int(np.log(rf) / np.log(ks))
        dilations = [ks**i for i in range(nl)]
        self.dilations = dilations + dilations   # dilation: 2^num_layers
        #self.dilations = [1] + [2] + [4] * 7  # costum dilations

        # classifier arguments
        self.wavenet_class = None  # class of wavenet model
        self.load_conv = 'y'  # where to load neural nerwork weights from
        self.pred = False  # whether to use wavenet in prediction mode
        self.init_model = True  # whether to reinitialize classifier
        self.reg_semb = True  # whether to regularize subject embedding
        self.fixed_wavenet = False  # whether to fix weights of wavenet
        self.alpha_norm = 0.0  # regularization multiplier on weights
        self.num_classes = 118  # number of classes for classification
        self.units = [2200, 2000]  # hidden layer sizes of fully-connected block
        self.dim_red = 80  # number of pca components for channel reduction
        self.stft_freq = 0  # STFT frequency index for LDA_wavelet_freq model
        self.decode_peak = 0.1
        self.trial_average = False

        # quantized wavenet arguments
        self.mu = 255
        self.residual_channels = 1024
        self.dilation_channels = 1024
        self.skip_channels = 1024
        self.class_emb = 10
        self.channel_emb = 30
        self.cond_channels = self.class_emb + self.channel_emb
        self.head_channels = int(self.skip_channels/2)
        self.conv_bias = False

        # dataset arguments
        data_path = os.path.join('/', 'gpfs2', 'well', 'woolrich', 'projects',
                                 'cichy118_cont', 'preproc_data_onepass', 'epoched')
        self.data_path = [os.path.join(data_path, f'subj{i}') for i in range(n)]  # path(s) to data directory
        self.num_channels = [-2] #[0, 1, 2, 64, 65, 66]  # channel indices
        self.numpy = True  # whether data is saved in numpy format
        self.crop = 1  # cropping ratio for trials
        self.shuffle = True
        self.whiten = False  # pca components used in whitening
        self.group_whiten = False  # whether to perform whitening at the GL
        self.split = np.array([0, 0.2])  # validation split (start, end)
        self.sr_data = 100  # sampling rate used for downsampling
        self.streaming_SR = 5
        self.original_sr = 1000
        self.save_data = True  # whether to save the created data
        self.save_whiten = False
        self.subjects_data = False  # list of subject inds to use in group data
        self.num_clip = 25
        self.dump_data = [os.path.join(data_path, f'subj{i}', 'train_data_trialnorm', 'c') for i in range(n)]  # path(s) for dumping data
        self.load_data = self.dump_data  # path(s) for loading data files

        # analysis arguments
        self.kernelPFI = False
        self.closest_chs = 'notebooks/closest1'  # channel neighbourhood size for spatial PFI
        self.PFI_inverse = False  # invert which channels/timesteps to shuffle
        self.pfich_timesteps = [[0, 50]]  # time window for spatiotemporal PFI
        self.PFI_perms = 10  # number of PFI permutations
        self.halfwin = 7  # half window size for temporal PFI
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
