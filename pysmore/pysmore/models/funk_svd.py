import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *


class funk_svd(torch.nn.Module):
    def __init__(self, features, mlp_params=None, **kwargs):
        super(funk_svd, self).__init__()

        self.features = features

        self.n_dims = sum([fea.embed_dim for fea in features])

        self.bias = nn.ParameterDict()

        for fea in features:
            # torch.manual_seed(0)
            self.bias[fea.name] = nn.Parameter(torch.randn(fea.vocab_size))
            # self.bias[fea.name] = nn.Parameter(torch.zeros(fea.vocab_size))

        self.bias['global_bias'] = nn.Parameter(torch.zeros(1))

        self.embedding = EmbeddingLayer(features,
                                        pretrain_embs=kwargs['pretrain_embs'],
                                        embed_init=kwargs['embed_init'])

    def debug(self):
        print(self.bias['F@u'].data)
        print(self.bias['F@i'].data)
        print(self.embedding.embed_dict['F@u'].weight.data)
        print(self.embedding.embed_dict['F@i'].weight.data)
        # print(self.bias['global_bias'])

    def forward(self, x):

        forward_features = [fea for fea in self.features if fea.name in x]

        sparse_emb, _ = self.embedding(x, forward_features, squeeze_dim=False)

        eu, ei = sparse_emb
        score = torch.mul(eu, ei).sum(dim=1)

        for fea in forward_features:
            score += self.bias[fea.name][x[fea.name]]

        score += self.bias['global_bias']

        score = torch.clamp(score, min=1, max=5)

        return score
