import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *

class fm(nn.Module):
    def __init__(self, features, features_split_size=None, mlp_params=None):
        super().__init__()
        """
        basic member
        """
        self.features = features
        self.n_dims = sum([fea.embed_dim for fea in self.features])

        """
        Model Arch
        """
        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.n_dims, 3)
        self.mlp = MLP(self.n_dims, output_layer=False, **mlp_params)
        self.linear = LinearLayer(self.n_dims + mlp_params["dims"][-1])
        

    def forward(self, x):
        emb_x = self.embedding(x, self.features, squeeze_dim=True)
        cn_out = self.cn(emb_x)
        mlp_out = self.mlp(emb_x)
        x_stack = torch.cat([cn_out, mlp_out], dim=1)
        y_pred = self.linear(x_stack)

        return torch.sigmoid(y_pred.squeeze(1))
