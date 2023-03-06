from torch.nn import Sequential, Conv1d, Module, CrossEntropyLoss, Embedding
from torch.nn import Softmax, Linear, Dropout, ModuleList, LayerNorm
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

# import savemat
from scipy.io import savemat

import os

from wavenets_full import WavenetFullChannelMix

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


class TransformerQuantized(GPT2Model):
    def __init__(self, args):
        super().__init__(args.gpt2_config)
        self.args = args
        self.build_model(args)

        self.criterion = CrossEntropyLoss(reduction='none').cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()

        self.post_init()

    def get_cond(self, *args, **kwargs):
        return WavenetFullChannelMix.get_cond(self, *args, **kwargs)

    def loss(self, *args, **kwargs):
        return WavenetFullChannelMix.loss(self, *args, **kwargs)

    def build_model(self, args):
        self.quant_levels = args.mu + 1
        self.save_preds = False

        # output head
        self.head = Linear(args.quant_emb, self.quant_levels, bias=False)

        # embeddings for various conditioning
        self.cond_emb = Embedding(args.num_classes, args.class_emb)

        # quantization embedding
        self.quant_emb = Embedding(self.quant_levels, args.quant_emb)

        # channel embeddings
        self.ch_emb = Embedding(args.num_channels, args.channel_emb)

        self.ch_ids = torch.arange(args.num_channels).cuda()

    def forward(self, data):
        x = data['inputs']  # B x C x T
        cond = self.get_cond(data)

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

        # concatenate all embeddings: B x C x T x (E_q + E_pos + E_ch + E_c)
        #x = torch.cat([x, pos_emb, ch_emb, cond], dim=-1)
        x = x + cond + ch_emb
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

        #outputs = outputs[0] @ self.quant_emb.weight.T
        outputs = self.head(outputs[0])

        # have to inspect output shape
        outputs = outputs.reshape(
            -1, self.args.num_channels, outputs.shape[1], outputs.shape[2])
        
        out_times = self.args.sample_rate - self.args.rf
        return outputs[:, :, -out_times-1:, :]

