import numpy as np

import torch
from torch.nn import Linear, ReLU, Module, MultiheadAttention, MSELoss


class AttentionLayer(Module):
    def __init__(self, emb_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.attn = MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)
        self.ffn1 = Linear(emb_dim, emb_dim*2)
        self.ffn2 = Linear(emb_dim*2, emb_dim)
        self.activation = ReLU()

    def forward(self, x, mask):
        x2 = self.attn(x, x, x, attn_mask=mask)[0]
        x = x + x2
        x2 = self.ffn2(self.activation(self.ffn1(x)))
        x = x + x2
        return x2


class ARAttention(Module):
    def __init__(self, seq_len):
        super(ARAttention, self).__init__()
        self.emb_dim = 32
        # might add ffn layer on top
        self.input_layer = Linear(1, int(self.emb_dim/2))
        self.attn1 = AttentionLayer(self.emb_dim, 4)
        self.attn2 = AttentionLayer(self.emb_dim, 4)
        self.attn3 = AttentionLayer(self.emb_dim, 4)
        self.attn4 = AttentionLayer(self.emb_dim, 4)

        self.mask = torch.triu(torch.ones((seq_len, seq_len),
                               dtype=torch.uint8))
        self.mask = self.mask - torch.diag(torch.ones((seq_len),
                                           dtype=torch.uint8))
        self.mask = self.mask.bool().cuda()

        remb = np.random.randn(seq_len, int(self.emb_dim/2))
        self.pos_emb = torch.tensor(remb, requires_grad=True).float().cuda()

        self.output_layer = Linear(self.emb_dim, 1)

        self.criterion = MSELoss().cuda()

    # Defining the forward pass
    def forward(self, x):
        batch_size = x.shape[1]
        x = self.input_layer(x)

        embeddings = torch.stack([self.pos_emb for i in range(batch_size)])
        embeddings = embeddings.permute(1, 0, 2)
        # x = embeddings + x
        x = torch.cat((embeddings, x), axis=2)

        x = self.attn1(x, self.mask)
        x = self.attn2(x, self.mask)
        x = self.attn3(x, self.mask)
        x = self.attn4(x, self.mask)

        x = self.output_layer(x)
        return x

    def loss(self, x):
        output = self.forward(x[:-1, :, :])
        loss = self.criterion(output, x[1:, :, :])
        return loss
