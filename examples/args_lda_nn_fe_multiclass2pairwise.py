import os
import torch
import torch.nn.functional as F
import numpy as np

from cichy_data import CichyData
from classifiers_linear import LDA


class Args:
    gpu = '1'  # cuda gpu index
    func = {'multi2pair': True}  # dict of functions to run from training.py

    def __init__(self):
        n = 1  # can be used to do multiple runs, e.g. over subjects

        # experiment arguments
        self.name = 'args.py'  # name of this file, don't change
        self.fix_seed = True  # whether to fix random seed
        self.load_model = [os.path.join(  # path(s) to load model
            'results_test',
            'lda_nn',
            'subj' + str(i),
            'model.pt25') for i in range(n)]
        self.result_dir = [os.path.join(  # path(s) to save model
            'results_test',
            'lda_nn',
            'subj' + str(i)) for i in range(n)]
        self.model = LDA  # class of model to use
        self.dataset = CichyData  # dataset class for loading and handling data
        self.max_trials = 1.0  # ratio of training data (1=max)

        # neural network arguments
        self.learning_rate = 0.00001  # learning rate for Adam
        self.batch_size = 3  # batch size for training and validation data
        self.epochs = 1000  # number of loops over training data
        self.val_freq = 10  # how often to validate (in epochs)
        self.print_freq = 2  # how often to print metrics (in epochs)
        self.units = [800, 300]  # hidden layer sizes of fully-connected block

        # classification arguments
        self.sample_rate = [10, 60]  # start and end of timesteps within trials
        # [10, 60] corresponds to 0 to 500 ms at 100 Hz
        self.num_classes = 118  # number of classes for classification
        self.dim_red = 80  # number of pca/learnt pca components
        # for channel reduction
        self.load_conv = [os.path.join(
            'results_test',
            'neural_network',
            'subj' + str(i),
            'model_end.pt') for i in range(n)]  # path to trained neural network from which
        # to extract the learnt pca layer
        self.halfwin = 25  # half window size for temporal PFI
        # and/or half window size for sliding window LDA

        # dataset arguments
        data_path = os.path.join('data', 'preproc')
        self.data_path = [os.path.join(data_path, 'subj' + str(i))
                          for i in range(n)]  # path(s) to data directory
        self.num_channels = list(range(307))  # channel indices
        # this is normally 306, but has to be 307 when loading data
        # with self.load_data
        self.numpy = True  # whether data is saved in numpy format
        self.crop = 1  # cropping ratio for trials (1 = no cropping)
        self.shuffle = False  # whether to shuffle trials
        self.whiten = False  # pca components used in whitening
        self.split = np.array([0, 0.2])  # validation split (start, end)
        self.sr_data = 100  # sampling rate used for downsampling
        self.original_sr = 1000  # original sampling rate
        self.save_data = True  # whether to save the created data
        self.dump_data = [os.path.join(d, 'data_files', 'c')
                          for d in self.data_path]  # path(s) for saving data
        self.load_data = self.dump_data  # path(s) for loading data files
        # once data has been created once just set load_data to dump_data
        # in subsequent runs on the same data

        # PFI arguments
        self.closest_chs = 20  # channel neighbourhood size for spatial PFI
        self.PFI_inverse = False  # corresponds to the inverse PFI mehtod
        # described in the paper
        self.pfich_timesteps = [0, 256]  # time window for spatiotemporal PFI
        self.PFI_perms = 20  # number of PFI permutations
        self.halfwin_uneven = False  # whether to use even or uneven window


        '''
        .
        .
        The following parameters are not used currently.
        .
        .
        '''
        self.activation = None  # activation function for models
        self.subjects = 0  # number of subjects used for training
        self.embedding_dim = 0  # subject embedding size
        self.p_drop = 0.0  # dropout probability
        self.ch_mult = 2  # channel multiplier for hidden channels in wavenet
        self.groups = 306
        self.kernel_size = 2  # convolutional kernel size
        self.timesteps = 1  # how many timesteps in the future to forecast

        self.val_max_trials = False
        self.subjects_data = False  # list of subject inds to use in group data
        self.save_whiten = False
        self.bypass = False
        self.group_whiten = False  # whether to perform whitening at the GL
        self.num_clip = 4
        self.rf = 128  # receptive field of wavenet, 2*rf - 1
        rf = 128
        ks = self.kernel_size
        nl = int(np.log(rf) / np.log(ks))
        dilations = [ks**i for i in range(nl)]
        self.dilations = dilations + dilations   # dilation: 2^num_layers
        #self.dilations = [1] + [2] + [4] * 7  # costum dilations

        self.generate_noise = 1  # noise used for wavenet generation
        self.generate_length = self.sr_data * 1000  # generated timeseries len
        self.generate_mode = 'recursive'  # IIR or FIR mode for wavenet generation
        self.generate_input = 'data'  # input type for generation
        self.generate_sampling = 'top-p'
        self.top_p = 0.8
        self.individual = True  # whether to analyse individual kernels
        self.anal_lr = 0.001  # learning rate for input backpropagation
        self.anal_epochs = 200  # number of epochs for input backpropagation
        self.norm_coeff = 0.0001  # L2 of input for input backpropagation
        self.kernel_limit = 300  # max number of kernels to analyse

        # classifier arguments
        self.wavenet_class = None  # class of wavenet model
        # dimensionality reduction from
        self.pred = False  # whether to use wavenet in prediction mode
        self.init_model = True  # whether to reinitialize classifier
        self.reg_semb = True  # whether to regularize subject embedding
        self.fixed_wavenet = False  # whether to fix weights of wavenet
        self.alpha_norm = 0.0  # regularization multiplier on weights
        self.stft_freq = 0  # STFT frequency index for LDA_wavelet_freq model
        self.decode_peak = 0.1

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
        self.order = 20
        self.uni = True
        self.save_AR = True
        self.do_anal = False
        self.AR_load_path = [os.path.join(  # path(s) to save model and others
            'results',
            'cichy_epoched',
            'subj1',
            'cont_quantized',
            'AR_uni')]

        # unused
        self.num_plot = 1
        self.plot_ch = 1
        self.linear = False
        self.num_samples_CPC = 20
        self.dropout2d_bad = False
        self.k_CPC = 1
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

        self.common_dataset = False
        self.load_dataset = True  # whether to load self.dataset
        self.anneal_lr = False  # whether to anneal learning rate
        self.save_curves = True  # whether to save loss curves to file
