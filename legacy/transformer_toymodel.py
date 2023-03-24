import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 seq_len,
                 buffer=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim,
                                               num_heads,
                                               batch_first=True)

        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Linear(embed_dim, embed_dim)

        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.mask = torch.triu(torch.ones(seq_len, seq_len),
                               diagonal=buffer).bool().cuda()

    def forward(self, x):
        out = self.attention(x, x, x, attn_mask=self.mask)[0]
        x = self.layer_norm1(x + out)

        out = self.feed_forward(x)
        x = self.layer_norm2(x + out)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 seq_len,
                 vocab_size,
                 num_layers,
                 num_heads=8,
                 buffer=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.layers = nn.ModuleList([
            Layer(embedding_dim, num_heads, seq_len, buffer)
            for _ in range(num_layers)])

        self.cross_entropy = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        # inputs: (N, L) -> (N, L, E)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        # get output probabilties by applying embedding weight transpose
        x = x @ self.embedding.weight.T

        return x

    def loss(self, inputs, targets):
        '''
        inputs: (N, L) batch x input sequence length,
            contains indices
            0 : the
            1 : dog
            2 : cat 
            ...
        targets: (N, K) batch x target sequence length,
            contains indices, like inputs
        '''
        outputs = self.forward(inputs)

        outputs = outputs[:, -targets.shape[1]-1:-1, :]
        loss = self.cross_entropy(outputs.reshape(-1, outputs.shape[-1]),
                                  targets.reshape(-1))

        return loss, outputs


def main():
    # batch size
    N = 2
    # input sequence length
    L = 8
    # embedding dimension
    E = 2
    # vocab size
    V = 10
    # number of layers
    NL = 2
    # number of heads
    H = 2

    inputs = torch.randint(V, (N, L)).cuda()
    targets = inputs[:, 1:].cuda()

    print(inputs)
    print(targets)

    model = Decoder(E, L, V, NL, H).cuda()

    # print model parameters
    for name, param in model.named_parameters():
        print(name, param.shape)

    print(model.layers[0].mask)

    loss, outputs = model.loss(inputs, targets)
    print(loss.item())
    print(outputs)

if __name__ == '__main__':
    main()