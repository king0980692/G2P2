import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
    from utils import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *
    from pysmore.utils import *


class lightGCN(nn.Module):

    def __init__(self, features, mlp_params=None, **kwargs):
        super(lightGCN, self).__init__()
        # assert len(features) == 3, "LightGCN only support 3 features(users, items, neg_items) now!"
        self.features = features
        self.n_users = features[0].vocab_size
        self.n_items = features[1].vocab_size 


        self.n_layers = 3

        """
        Create the adjacency matrix and calculate the norm matrix
        """
        # self.adj_mat = self.create_sp_mat(kwargs['inter_df'])
        # self.norm_adj_mat_sparse_tensor = self.get_A_tilda()

        self.embedding = lightGCN_Embedding(kwargs['inter_df'],
                                            features,
                                            n_layers=self.n_layers,
                                            pretrain_embs= \
                                            kwargs['pretrain_embs']
                                            )



    def forward(self, x):
        forward_features = [fea for fea in self.features if fea.name in x]
        
        sparse_emb = self.embedding(x, forward_features)

        eu, ei = sparse_emb

        # users_emb, pos_emb, neg_emb = out_u_emb[users], out_i_emb[pos_items], out_i_emb[neg_items]
        # userEmb0,  posEmb0, negEmb0 = init_user_emb[users], init_item_emb[pos_items], init_item_emb[neg_items]

        # return users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0

        score = torch.mul(eu, ei).sum(dim=1)

        return score

