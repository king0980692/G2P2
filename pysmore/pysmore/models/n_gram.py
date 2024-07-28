import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *


class n_gram(torch.nn.Module):
    def __init__(self, features, mlp_params=None, **kwargs):
        super(n_gram, self).__init__()

        self.features = features

        self.embedding = EmbeddingLayer(
            features, pretrain_embs=kwargs['pretrain_embs'])

    def forward(self, x):

        forward_features = [fea for fea in self.features if fea.name in x]

        sparse_emb, _ = self.embedding(x, forward_features, squeeze_dim=False)

        eu, ei = sparse_emb

        score = torch.mul(eu, ei).sum(dim=1)

        return score
