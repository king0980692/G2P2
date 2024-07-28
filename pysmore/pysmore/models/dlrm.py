import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *

class dlrm(nn.Module):
    def __init__(self, features, mlp_params=None, **kwargs):
        super().__init__()

        self.mlp_params = mlp_params
        self.features = features

        self.n_dense = len([fea \
                            for fea in self.features \
                            if isinstance(fea, DenseFeature)  and 'neg' not in fea.name])

        self.n_sparse = len([fea \
                            for fea in self.features \
                            if isinstance(fea, SparseFeature)  and 'neg' not in fea.name])

        """
        Model Arch
        """

        self.bot_mlp = MLP(self.n_dense, 
                           dims=mlp_params["bot_dims"],
                           dropout=mlp_params["dropout"])

        if self.n_dense > 0 :
            top_dims = ( self.n_sparse * (self.n_sparse + 1) ) // 2  # 1 means the dense_fea
            top_dims += mlp_params["bot_dims"][-1] # concat the dense at the end of interactive stage
        else:
            top_dims = ( self.n_sparse * (self.n_sparse - 1) ) // 2  # 2-order of sparse feature

        self.top_mlp = MLP(top_dims, 
                           output_layer=kwargs['point_output'],
                           dims=mlp_params["top_dims"],
                           bias=False)


        self.embedding = EmbeddingLayer(features, self.bot_mlp)
        
    def interact_features(self, x, ly, mode=None):
        """
        x : dense_feature
        ly: list of sparse_feature
        """
        if not isinstance(x, list):
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))

            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2)) # AA^T matrix

            _, ni, nj = Z.shape
            offset = -1

            li, lj = torch.tril_indices(ni, nj, offset=offset)

            Zflat = Z[:, li, lj]

            R = torch.cat([x] + [Zflat], dim=1)

        else:
            """
            Only sparse feature case
            """

            batch_size = ly[0].shape[0]
            
            d =ly[0].shape[1]
 
            T = torch.cat(ly, dim=1).view((batch_size, -1, d))

            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2)) # AA^T matrix

            _, ni, nj = Z.shape
            offset = -1

            li, lj = torch.tril_indices(ni, nj, offset= offset)

            Zflat = Z[:, li, lj]

            R = torch.cat([Zflat], dim=1).sum(dim=1)

        return R

    def forward(self, x):

        forward_features = [fea for fea in self.features if fea.name in x]
        
        emb_x, dense_val = self.embedding(x, forward_features, squeeze_dim=False) 

        p = self.interact_features(dense_val, emb_x)

        logits = self.top_mlp(p)
        # .squeeze(1)

        return logits.squeeze(-1)
        # return torch.sigmoid(logits) # prediction

