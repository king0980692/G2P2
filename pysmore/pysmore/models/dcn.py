import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *

class dcn(nn.Module):
    def __init__(self, features, mlp_params=None, **kwargs):
        super(dcn, self).__init__()

        """
        basic member
        """
        self.features = features
        self.n_dims = sum([fea.embed_dim for fea in self.features if 'neg' not in fea.name])

        """
        Model Arch
        """

        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.n_dims, 3)
        self.mlp = MLP(self.n_dims, output_layer=False, dims=mlp_params['top_dims'])
        self.linear = LinearLayer(self.n_dims + mlp_params["top_dims"][-1])
        


    def forward(self, x):

        forward_features = [fea for fea in self.features if fea.name in x]
        emb_x = self.embedding(x, forward_features, squeeze_dim=True)

        cn_out = self.cn(emb_x)
        mlp_out = self.mlp(emb_x)
        x_stack = torch.cat([cn_out, mlp_out], dim=1)
        y_pred = self.linear(x_stack)
        
        return torch.sigmoid(y_pred.squeeze(1))
