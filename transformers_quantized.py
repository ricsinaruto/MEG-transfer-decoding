from torch.nn import CrossEntropyLoss, Embedding, Linear
from torch.nn.parameter import Parameter
import torch

from wavenets_full import WavenetFullChannelMixMixin, WavenetFullTest

from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class TransformerQuantized(GPT2Model, WavenetFullChannelMixMixin):
    def __init__(self, args):
        super().__init__(args.gpt2_config)
        self.args = args
        self.build_model(args)

        self.criterion = CrossEntropyLoss(reduction='none').cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()

        self.post_init()

    def loaded(self, args):
        self.args = args

    def generate(self, train_data):
        self.out_times = 0
        return super().generate(train_data)

    def build_model(self, args):
        self.quant_levels = args.mu + 1
        self.out_times = args.sample_rate - args.rf
        self.save_preds = False

        # output head
        self.head = Linear(args.gpt2_config.n_embd,
                           self.quant_levels,
                           bias=False)

        # embeddings for various conditioning
        self.cond_emb = Embedding(args.num_classes, args.class_emb)

        # quantization embedding
        self.quant_emb = Embedding(self.quant_levels, args.quant_emb)

        # channel embeddings
        self.ch_emb = Embedding(args.num_channels, args.channel_emb)

        self.ch_ids = torch.arange(args.num_channels).cuda()

    def embedding_method(self, x, cond, ch_emb):
        return x + cond + ch_emb

    def forward_head(self, x):
        # outputs = outputs[0] @ self.quant_emb.weight.T
        x = self.head(x)
        # have to inspect output shape
        x = x.reshape(-1, self.args.num_channels, x.shape[1], x.shape[2])

        return x[:, :, -self.out_times-1:, :]

    def forward(self, data):
        x = data['inputs']  # B x C x T
        cond = self.get_cond(data)
        if cond:
            # reshape to B x C x T x E_c
            cond = cond.reshape(
                -1, self.args.num_channels, cond.shape[1], cond.shape[2])
            cond = cond.permute(0, 1, 3, 2)

        timesteps = x.shape[-1]

        # apply quantization embedding
        x = self.quant_emb(x)  # B x C x T x E_q

        # get channel embedding: C x E_ch
        ch_emb = self.ch_emb(self.ch_ids).unsqueeze(0)
        # repeat across batch and time:  B x T x C x E_ch
        ch_emb = ch_emb.repeat(timesteps, 1, 1).unsqueeze(0)
        ch_emb = ch_emb.permute(0, 2, 1, 3)  # B x C x T x E_ch

        # get positional embedding: T x E_pos
        '''
        pos_emb = self.pos_emb(torch.arange(timesteps).to(x.device))
        # repeat across batch and channels
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(2).repeat(
            x.shape[0], 1, self.args.num_channels, 1)
        '''

        # add up embeddings
        x = self.embedding_method(x, cond, ch_emb)
        x = x.reshape(-1, timesteps, x.shape[-1])  # B*C x T x E

        outputs = super().forward(input_ids=None,
                                  past_key_values=None,
                                  attention_mask=None,
                                  token_type_ids=None,
                                  position_ids=None,
                                  head_mask=None,
                                  inputs_embeds=x,
                                  encoder_hidden_states=None,
                                  encoder_attention_mask=None,
                                  use_cache=None,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None)

        outputs = self.forward_head(outputs[0])

        return outputs


class TransformerQuantizedConcatEmb(TransformerQuantized):
    def embedding_method(self, x, cond, ch_emb):
        return torch.cat([x, cond, ch_emb], dim=-1)


class TransformerQuantizedConcatOut(TransformerQuantized):
    '''
    Expands on the TransformerQuantizedConcatEmb by concatenating the output
    across the channel dimension, and then applying a separate
    linear layer (head) to predict the output of each channel.
    '''
    def build_model(self, args):
        super().build_model(args)

        # output head
        shape = (args.num_channels,
                 args.gpt2_config.n_embd * args.num_channels,
                 args.quant_emb)
        self.head = torch.normal(size=shape,
                                 dtype=torch.float32,
                                 requires_grad=True,
                                 device='cuda',
                                 mean=0.0,
                                 std=self.config.initializer_range)
        self.head = Parameter(self.head)

        self.head2 = Linear(args.quant_emb, self.quant_levels, bias=False)

    def forward_head(self, x):
        # reshape to B x C x T x E
        x = x.reshape(-1, self.args.num_channels, x.shape[1], x.shape[2])
        x = x[:, :, -self.out_times-1:, :]

        # join embedding and channel dimension
        x = x.permute(0, 2, 1, 3)  # B x T x C x E
        x = x.reshape(x.shape[0], x.shape[1], -1)  # B x T x C*E

        # apply each channel's head
        # B x T x C x E
        x = torch.tensordot(x, self.head, dims=([2], [1]))  # type: ignore
        x = self.head2(x)  # B x T x C x Q
        return x.permute(0, 2, 1, 3)  # B x C x T x Q


class TransformerQuantizedDiffCH(TransformerQuantizedConcatEmb):
    def embedding_method(self, x, cond):
        return torch.cat([x, cond], dim=-1)

    def build_model(self, args):
        super().build_model(args)

        # different quant embedding for each channel
        self.quant_emb = torch.randn(
            size=(args.num_channels, self.quant_levels, args.quant_emb),
            dtype=torch.float32,
            requires_grad=True,
            device='cuda')
        self.quant_emb = Parameter(self.quant_emb)


class TransformerQuantizedChMix(TransformerQuantized):
    def embedding_method(self, x, cond):
        return torch.cat([x, cond], dim=-1)

    def get_cond(self, *args, **kwargs):
        return WavenetFullTest.get_cond(self, *args, **kwargs)

    def build_model(self, args):
        self.quant_levels = args.mu + 1
        self.save_preds = False

        # output head
        shape = (args.num_channels, args.gpt2_config.n_embd, self.quant_levels)
        self.head = torch.normal(size=shape,
                                 dtype=torch.float32,
                                 requires_grad=True,
                                 device='cuda',
                                 mean=0.0,
                                 std=self.config.initializer_range)
        self.head = Parameter(self.head)

        # embeddings for various conditioning
        self.cond_emb = Embedding(args.num_classes, args.class_emb)

        # quantization embedding
        self.quant_emb = Embedding(self.quant_levels, args.quant_emb)

        self.ch_ids = torch.arange(args.num_channels).cuda()

    def forward(self, data):
        x = data['inputs']  # B x C x T
        bs = x.shape[0]
        ts = x.shape[-1]

        # apply quantization embedding
        x = self.quant_emb(x)  # B x C x T x E_q
        x = x.permute(0, 2, 1, 3).reshape(bs, ts, -1)  # B x T x C*E_q

        cond = self.get_cond(data)
        if cond:
            cond = cond.permute(0, 2, 1)  # B x T x E_c

        # add up embeddings
        x = self.embedding_method(x, cond)  # B x T x (C*E_q + E_c)

        x = GPT2Model.forward(self,
                              input_ids=None,
                              past_key_values=None,
                              attention_mask=None,
                              token_type_ids=None,
                              position_ids=None,
                              head_mask=None,
                              inputs_embeds=torch.FloatTensor(x),
                              encoder_hidden_states=None,
                              encoder_attention_mask=None,
                              use_cache=None,
                              output_attentions=None,
                              output_hidden_states=None,
                              return_dict=None)

        out_times = self.args.sample_rate - self.args.rf
        x = x[0][:, -out_times-1:, :]

        # apply each channel's head
        # outputs = torch.einsum('ijl,klq->ijkq', outputs[0], self.head)

        x = torch.tensordot(x, self.head, dims=([2], [1]))  # type: ignore
        return x.permute(0, 2, 1, 3)  # B x C x T x Q


class TransformerQuantizedChMixSmall(TransformerQuantizedChMix):
    def build_model(self, args):
        super().build_model(args)

        # output head
        shape = (args.num_channels, args.gpt2_config.n_embd, args.quant_emb)
        head = torch.normal(size=shape,
                            dtype=torch.float32,
                            requires_grad=True,
                            device='cuda',
                            mean=0.0,
                            std=self.config.initializer_range)
        self.head = Parameter(head)

        self.head2 = Linear(args.quant_emb, self.quant_levels, bias=False)

    def forward(self, data):
        x = super().forward(data)

        # apply head2
        return self.head2(x)
