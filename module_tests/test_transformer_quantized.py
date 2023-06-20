import os

os.environ["NVIDIA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Generated by CodiumAI
import pytest
from transformers_quantized import TransformerQuantized
from args_test import Args
from torch.nn import Linear, Embedding, CrossEntropyLoss, MSELoss
import torch

from transformers import GPT2Config

"""
Code Analysis:
- The class 'TransformerQuantized' is a subclass of 'GPT2Model' and 'WavenetFullChannelMix' classes, which inherits their properties and methods.
- It is used for generating audio samples from a trained model.
- The 'build_model' method initializes the model's parameters and embeddings, including the quantization embedding, channel embedding, and positional embedding.
- The 'embedding_method' method adds up the embeddings for each input sample.
- The 'forward_head' method applies the linear output head to the model's output and reshapes it to the desired output shape.
- The 'forward' method applies the quantization embedding, channel embedding, and positional embedding to the input data, and then passes it through the GPT2 model. The output is then passed through the output head to get the final output.
- The 'get_cond' method retrieves the conditioning data for the model.
- The 'generate' method generates audio samples using the trained model and the given input data.
- The 'criterion' and 'mse_loss' fields define the loss functions used for training the model.
- The 'quant_levels', 'out_times', and 'save_preds' fields define the parameters for generating audio samples.
"""


@pytest.fixture
def args_test():
    args = Args()

    args.mu = 255
    args.sample_rate = 256
    args.rf = 128
    args.num_classes = 10

    n_embd = 12*8
    args.gpt2_config = GPT2Config(
        vocab_size=256,
        n_positions=256,
        n_embd=n_embd,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=255,
        eos_token_id=255,
        name_or_path=None,
        use_cache=False
    )

    args.class_emb = n_embd
    args.quant_emb = n_embd
    args.num_channels = 4
    args.channel_emb = n_embd

    args.cond_channels = args.class_emb + args.embedding_dim
    
    args.result_dir = os.path.join(
            '/',
            'well',
            'woolrich',
            'users',
            'yaq921',
            'MEG-transfer-decoding',  # path(s) to save model and others
            'module_tests',
            'outputs')
    return args


def devices(*devices_):
    return pytest.mark.parametrize(
        "device", [torch.device(device) for device in devices_], ids=devices_
    )


cpu = devices("cpu")
cuda = devices("cuda")
cpu_and_cuda = devices("cpu", "cuda")


class TestTransformerQuantized:

    # Tests that the model's parameters and embeddings are initialized properly. tags: [happy path]
    def test_build_model(self, args_test):
        args = args_test

        model = TransformerQuantized(args).cuda()
        assert model.quant_levels == args.mu + 1
        assert model.out_times == args.sample_rate - args.rf
        assert model.save_preds == False
        assert isinstance(model.head, Linear)
        assert isinstance(model.cond_emb, Embedding)
        assert isinstance(model.quant_emb, Embedding)
        assert isinstance(model.ch_emb, Embedding)
        assert model.ch_ids.tolist() == [0, 1, 2, 3]

    # Tests that the model applies quantization embedding, channel embedding, and positional embedding to input data and generates the desired output shape. tags: [happy path]
    def test_forward(self, args_test):
        args = args_test

        model = TransformerQuantized(args).cuda()
        data = {'inputs': torch.randint(args.mu+1, size=(2, 4, 256)).cuda()}

        # define condition
        data['condition'] = torch.randint(2, size=(2, 1, 256)).cuda()
        output = model(data)
        assert output.shape == (2, 4, args.sample_rate - args.rf + 1, args.mu + 1)

    '''
    # Tests that the method generates audio samples using the trained model and the given input data. tags: [edge case]
    def test_generate(self, args_test):
        args = args_test
        
        model = TransformerQuantized(args).cuda()
        train_data = torch.randint(args.mu+1, size=(1000, 4, 256)).cuda()
        cond = torch.randint(2, size=(1000, 1, 256)).cuda()
        train_data = torch.concat((train_data, cond, cond), dim=1)

        output = model.generate(train_data)
        assert output.shape == (args.num_channels, args.generate_length + args.rf)
    '''

    # Tests that the method retrieves the conditioning data for the model properly. tags: [happy path]
    def test_get_cond(self, args_test):
        args = args_test
        
        model = TransformerQuantized(args).cuda()
        data = {'inputs': torch.randint(args.mu+1, size=(2, 4, 256)).cuda(),
                'condition': torch.randint(2, size=(2, 1, 256)).cuda()}
        cond = model.get_cond(data)
        assert cond.shape == (2 * args.num_channels, args.class_emb, 256)
        assert cond.sum() != 0

    # Tests that the method applies the linear output head to the model's output and reshapes it to the desired output shape. tags: [happy path]
    def test_forward_head(self, args_test):
        args = args_test
        model = TransformerQuantized(args).cuda()
        x = torch.randn(2 * args.num_channels, (args.sample_rate - args.rf)*2, args.gpt2_config.n_embd).cuda()
        output = model.forward_head(x)
        assert output.shape == (2, args.num_channels, args.sample_rate - args.rf + 1, args.mu + 1)

    # Tests that the loss functions used for training the model are defined properly. tags: [happy path]
    def test_loss_functions(self, args_test):
        args = args_test
        model = TransformerQuantized(args).cuda()
        criterion = model.criterion
        mse_loss = model.mse_loss
        assert isinstance(criterion, CrossEntropyLoss)
        assert isinstance(mse_loss, MSELoss)
