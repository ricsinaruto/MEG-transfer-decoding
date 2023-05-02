import os
import torch
import numpy as np
from transformers import GPT2Config

from gpt_quantized import GPT2Flat_masked
from cichy_data import CichyProductQuantized


class Args:
    gpu = '1'  # cuda gpu index
    func = {'generate': True}  # dict of functions to run from training.py

    def __init__(self):
        n = 1  # can be used to do multiple runs, e.g. over subjects

        # experiment arguments
        self.name = 'args.py'  # name of this file, don't change
        self.fix_seed = True
        self.common_dataset = False
        self.load_dataset = True  # whether to load self.dataset
        self.learning_rate = 0.0001  # learning rate for Adam
        self.max_trials = 1.0  # ratio of training data (1=max)
        self.val_max_trials = False
        self.batch_size = 1  # batch size for training and validation data
        self.epochs = 1000  # number of loops over training data
        self.val_freq = 2  # how often to validate (in epochs)
        self.print_freq = 1  # how often to print metrics (in epochs)
        self.anneal_lr = False  # whether to anneal learning rate
        self.save_curves = True  # whether to save loss curves to file
        self.load_model = [os.path.join(
            '/',
            'well',
            'woolrich',
            'users',
            'yaq921',
            'MEG-transfer-decoding',  # path(s) to save model and others
            'results',
            'cichy_epoched',
            'subj1',
            'cont_quantized',
            'GPT2Flat_masked')]
        self.result_dir = [os.path.join(
            '/',
            'well',
            'woolrich',
            'users',
            'yaq921',
            'MEG-transfer-decoding',  # path(s) to save model and others
            'results',
            'cichy_epoched',
            'subj1',
            'cont_quantized',
            'GPT2Flat_masked')]
        self.model = GPT2Flat_masked  # class of model to use
        self.dataset = CichyProductQuantized  # dataset class for loading and handling data

        # wavenet arguments
        self.activation = torch.nn.Identity()  # activation function for models
        self.subjects = 0  # number of subjects used for training
        self.embedding_dim = 0  # subject embedding size
        self.p_drop = 0.0  # dropout probability
        self.ch_mult = 2  # channel multiplier for hidden channels in wavenet
        self.groups = 306
        self.kernel_size = 2  # convolutional kernel size
        self.timesteps = 1  # how many timesteps in the future to forecast
        self.sample_rate = [0, 200]  # start and end of timesteps within trials
        self.example_shift = 100
        self.rf = 128  # receptive field of wavenet, 2*rf - 1
        rf = 128
        ks = self.kernel_size
        nl = int(np.log(rf) / np.log(ks))
        dilations = [ks**i for i in range(nl)]
        self.dilations = dilations + dilations   # dilation: 2^num_layers
        #self.dilations = [1] + [2] + [4] * 7  # costum dilations

        # classifier arguments
        self.wavenet_class = None  # class of wavenet model
        self.load_conv = False  # where to load neural nerwork
        # dimensionality reduction from
        self.pred = False  # whether to use wavenet in prediction mode
        self.init_model = True  # whether to reinitialize classifier
        self.reg_semb = True  # whether to regularize subject embedding
        self.fixed_wavenet = False  # whether to fix weights of wavenet
        self.alpha_norm = 0.0  # regularization multiplier on weights
        self.num_classes = 119  # number of classes for classification
        self.units = [800, 300]  # hidden layer sizes of fully-connected block
        self.dim_red = 16  # number of pca components for channel reduction
        self.stft_freq = 0  # STFT frequency index for LDA_wavelet_freq model
        self.decode_peak = 0.1

        # GPT2 arguments
        n_embd = 12*8
        self.gpt2_config = GPT2Config(
            vocab_size=16384,
            n_positions=240 * 31,
            n_embd=n_embd,
            n_layer=8,
            n_head=8,
            resid_pdrop=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
            bos_token_id=255,
            eos_token_id=255,
            name_or_path=None,
            use_cache=False
        )
        self.gpt2_config.num_channels = 31

        # quantized wavenet arguments
        self.skips_shift = 1
        self.mu = 255
        self.residual_channels = 128
        self.dilation_channels = 128
        self.skip_channels = 512
        self.channel_emb = n_embd
        self.ts_emb = n_embd
        self.class_emb = n_embd
        self.quant_emb = n_embd
        self.pos_emb = n_embd
        self.cond_channels = self.class_emb + self.embedding_dim
        self.head_channels = 256
        self.conv_bias = False

        # dataset arguments
        data_path = os.path.join('/', 'gpfs2', 'well', 'woolrich', 'projects',
                                 'cichy118_cont', 'preproc_data_osl', 'subj1')
        self.data_path = [[os.path.join(data_path, 'subj1_50hz.npy')]]  # path(s) to data directory
        self.num_channels = list(range(32))  # channel indices
        self.num_buckets = 30
        self.num_bits = 14
        self.numpy = True  # whether data is saved in numpy format
        self.crop = 1  # cropping ratio for trials
        self.whiten = False  # pca components used in whitening
        self.filter = None
        self.group_whiten = False  # whether to perform whitening at the GL
        self.split = np.array([0, 0.1])  # validation split (start, end)
        self.sr_data = 100  # sampling rate used for downsampling
        self.original_sr = 1000
        self.save_data = True  # whether to save the created data
        self.bypass = False
        self.subjects_data = False  # list of subject inds to use in group data
        self.save_whiten = False
        self.num_clip = 4
        self.dump_data = [os.path.join(data_path, '50hz100hz_productquantized')]  # path(s) for dumping data
        self.load_data = self.dump_data  # path(s) for loading data files

        # analysis arguments
        self.closest_chs = 20  # channel neighbourhood size for spatial PFI
        self.PFI_inverse = False  # invert which channels/timesteps to shuffle
        self.pfich_timesteps = [0, 256]  # time window for spatiotemporal PFI
        self.PFI_perms = 20  # number of PFI permutations
        self.halfwin = 5  # half window size for temporal PFI
        self.halfwin_uneven = False  # whether to use even or uneven window
        self.generate_noise = 1  # noise used for wavenet generation
        self.generate_length = self.sr_data * 2  # generated timeseries len
        self.generate_shift = 10
        self.generate_mode = 'recursive'  # IIR or FIR mode for wavenet generation
        self.generate_input = 'data'  # input type for generation
        self.generate_sampling = 'top-p'
        self.top_p = 0.8
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
