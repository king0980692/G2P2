import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *

class emb_cls(nn.Module):
    def __init__(self, features, mlp_params=None, **kwargs):
        super(emb_cls, self).__init__()
        self.features = features
        self.n_dims = sum([fea.embed_dim for fea in features])

        self.embedding = EmbeddingLayer(features)
        self.fc = MLP(output_layer=kwargs['point_output'],
                      num_labels=kwargs['n_classes'],
                      input_dim=self.n_dims)

    def forward(self, x):
        embedded = self.embedding(x, self.features, squeeze_dim=True)
        p = self.fc(embedded)

        return p.squeeze(1)
