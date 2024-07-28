import math
import torch.nn as nn
import torch
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0 # check the h number
        self.d_k = d_embed//h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0) # get batch size
        # 1) Linear projections to get the multi-head query, key and value tensors
        # x_query, x_key, x_value dimension: nbatch * seq_len * d_embed
        # LHS query, key, value dimensions: nbatch * h * seq_len * d_k
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key   = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        # 2) Attention
        # scores has dimensions: nbatch * h * seq_len * seq_len
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        # 3) Mask out padding tokens and future tokens
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        # p_atten dimensions: nbatch * h * seq_len * seq_len
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        # x dimensions: nbatch * h * seq_len * d_k
        x = torch.matmul(p_atten, value)
        # x now has dimensions:nbtach * seq_len * d_embed
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        return self.linear(x) # final linear layer


class ResidualConnection(nn.Module):
  '''residual connection: x + dropout(sublayer(layernorm(x))) '''
  def __init__(self, dim, dropout):
      super().__init__()
      self.drop = nn.Dropout(dropout)
      self.norm = nn.LayerNorm(dim)

  def forward(self, x, sublayer):
      return x + self.drop(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    '''EncoderBlock: self-attention -> position-wise fully connected feed-forward layer'''
    # def __init__(self, config):
    def __init__(self, features, mlp_params=None, **kwargs):
        super(EncoderBlock, self).__init__()

        embed_dim = features[0].embed_dim
        self.atten = MultiHeadedAttention(1, embed_dim, 0.1)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.residual1 = ResidualConnection(embed_dim, 0.1)
        self.residual2 = ResidualConnection(embed_dim, 0.1 )

    def forward(self, x, mask=None):
        # self-attention
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        # position-wise fully connected feed-forward layer
        return self.residual2(x, self.feed_forward)

class Encoder(nn.Module):
    '''Encoder = token embedding + positional embedding -> a stack of N EncoderBlock -> layer norm'''
    # def __init__(self, config):
    # def __init__(self, features, d_embed, vocab_size, N_encoder,dropout):
    def __init__(self, features, mlp_params=None, **kwargs):
        super().__init__()

        self.features = features
        self.d_embed = features[0].embed_dim
        self.tok_embed = EmbeddingLayer(features)
        # self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, kwargs['context_length'], self.d_embed))

        self.encoder_blocks = nn.ModuleList([EncoderBlock(features, mlp_params=None, **kwargs) for _ in range(1)])

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(self.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input, self.features, squeeze_dim=True)

        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)

class transformer(nn.Module):
    # def __init__(self, config, num_classes):
    def __init__(self, features, mlp_params=None, **kwargs):
        super().__init__()
        embed_dim = features[0].embed_dim
        self.encoder = Encoder(features, mlp_params=None, **kwargs)
        self.linear = nn.Linear(embed_dim, kwargs['n_classes'])

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        return  self.linear(torch.mean(x,-2))
