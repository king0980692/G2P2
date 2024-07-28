import scipy.sparse as sp
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .activation import activation_layer
from .initializers import smore_initializer, RandomInitializer, random_initializer


try:
    from pysmore.input.features import SparseFeature, SequenceFeature, DenseFeature
    from pysmore.utils import *
except:
    from input.features import SparseFeature, SequenceFeature, DenseFeature
    from utils import *


class lightGCN_Embedding(nn.Module):
    """lightGCN Embedding.
    Concat the whole Sparse feature to create the lightGCN embedding table.
    """

    def __init__(self,
                 inter_df,
                 features,
                 n_layers=3,
                 **kwargs
                 ):

        super().__init__()
        self.inter_df = inter_df
        self.features = features
        self.n_layers = n_layers
        self.n_users = self.inter_df.iloc[:, 0].max() + 1
        self.n_items = self.inter_df.iloc[:, 1].max() + 1

        if 'pretrain_embs' in kwargs:
            pretrain_embs = kwargs['pretrain_embs']
        else:
            pretrain_embs = [None]*len(features)

        all_sparse_features = [fea for fea in features
                               if isinstance(fea, SparseFeature)
                               and fea.shared_with == None]
        mat_size = sum([fea.vocab_size for fea in all_sparse_features])
        emb_dim = all_sparse_features[0].embed_dim

        self.E0 = nn.Embedding(mat_size, emb_dim)
        # nn.init.xavier_uniform_(self.E0.weight)

        self.norm_adj_mat_sp_tensor = self.get_A_tilda()

        """
        Preserve for every time update the embedding
        """
        self.embed_dict = nn.ModuleDict()

    def _create_sp_mat(self):
        """
        Note:
        Assume the inter_df has 3 columns: user, item, label

        """
        inter_df = self.inter_df
        R, adj = utils.create_sp_adj_mat(inter_df.iloc[:, 0],
                                         inter_df.iloc[:, 1],
                                         self.n_users,
                                         self.n_items)

        return adj

    def get_A_tilda(self):

        adj_mat = self._create_sp_mat()

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)

        # Below Code is toconvert the dok_matrix to sparse tensor.

        norm_adj_mat_coo = norm_adj_mat.tocoo().astype(np.float32)
        values = norm_adj_mat_coo.data
        indices = np.vstack((norm_adj_mat_coo.row, norm_adj_mat_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = norm_adj_mat_coo.shape

        norm_adj_mat_sp_tensor = torch.sparse.FloatTensor(
            i, v, torch.Size(shape))

        return norm_adj_mat_sp_tensor

    def propagate_through_layers(self, x, features):

        device = x[features[0].name].device

        E_lyr = self.E0.weight.to(device)
        all_layer_embedding = [self.E0.weight]

        self.norm_adj_mat_sp_tensor = self.norm_adj_mat_sp_tensor.to(device)

        for layer in range(self.n_layers):
            E_lyr = torch.sparse.mm(self.norm_adj_mat_sp_tensor, E_lyr)
            all_layer_embedding.append(E_lyr)

        all_layer_embedding = torch.stack(all_layer_embedding)
        mean_layer_embedding = torch.mean(all_layer_embedding, axis=0)

        out_embs = torch.split(mean_layer_embedding, [
                               self.n_users, self.n_items])

        """
        prepare emb_dict for forwarding !
        """
        for id, fea in enumerate(features):

            if isinstance(fea, SparseFeature):
                self.embed_dict[fea.name] = nn.Embedding(
                    fea.vocab_size, fea.embed_dim)
                self.embed_dict[fea.name].weight = nn.Parameter(
                    out_embs[id].clone())

    def forward(self,
                x,
                features):
        """
        Return List of embedding
        """

        # if neg forward pass, then no need to propagate
        if not any(['neg' in fea.name for fea in features]):
            self.propagate_through_layers(x, features)

        # -----  Forwarding -----
        sparse_emb = []
        for fea in features:
            fea_name = fea.name
            fea_key = fea.shared_with if fea.shared_with is not None \
                else fea.name
            if isinstance(fea, SparseFeature):
                sparse_emb.append(
                    self.embed_dict[fea_key](x[fea_name].long())
                )

        return sparse_emb

        # init_u_emb, init_i_emb = torch.split(self.E0.weight, [self.n_users, self.n_items])
        # return out_u_emb, out_i_emb, init_u_emb, init_i_emb


class EmbeddingLayer(nn.Module):
    """General Embedding Layer.
    We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.


    Args:
        features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.

    Shape:
        - Input:
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output:
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
    """

    def __init__(self,
                 features,
                 mlp=None,
                 **kwargs
                 ):

        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.mlp = mlp
        if 'pretrain_embs' in kwargs:
            pretrain_embs = kwargs['pretrain_embs']
        else:
            pretrain_embs = [None]*len(features)

        if 'embed_init' not in kwargs:
            kwargs['embed_init'] = 'smore'

        if kwargs['embed_init'] == 'rand':
            def emb_init_method(embedding_layer): return random_initializer(
                embedding_layer.weight)
        elif kwargs['embed_init'] == 'normal':
            def emb_init_method(embedding_layer, mean=0.0, std=1.0): return nn.init.normal_(
                embedding_layer.weight, mean=mean, std=std)
        elif kwargs['embed_init'] == 'uniform':
            def emb_init_method(embedding_layer, a=0.0, b=1.0): return nn.init.uniform_(
                embedding_layer.weight, a=a, b=b)
        elif kwargs['embed_init'] == 'xavier_normal':
            def emb_init_method(embedding_layer): return nn.init.xavier_normal_(
                embedding_layer.weight)
        elif kwargs['embed_init'] == 'xavier_uniform':
            def emb_init_method(embedding_layer): return nn.init.xavier_uniform_(
                embedding_layer.weight)
        elif kwargs['embed_init'] == 'kaiming_normal':
            def emb_init_method(embedding_layer): return nn.init.xavier_normal_(
                embedding_layer.weight)
        elif kwargs['embed_init'] == 'kaiming_uniform':
            def emb_init_method(embedding_layer): return nn.init.xavier_uniform_(
                embedding_layer.weight)
        else:  # default using smore init
            def emb_init_method(embedding_layer): return smore_initializer(
                embedding_layer.weight)

        for id, fea in enumerate(features):
            if fea.name in self.embed_dict:  # exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                emb_table = nn.Embedding(fea.vocab_size, fea.embed_dim)
                emb_init_method(emb_table)

                if pretrain_embs[id] is not None:
                    emb_table.weight.data.copy_(pretrain_embs[id])

                self.embed_dict[fea.name] = emb_table
            elif isinstance(fea, SequenceFeature):
                emb_table = nn.Embedding(fea.vocab_size, fea.embed_dim)
                self.embed_dict[fea.name] = emb_table

    def forward(self,
                x,
                features,
                squeeze_dim=False):

        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False

        # -----  Forwarding -----
        for fea in features:
            fea_name = fea.name
            fea_key = fea.shared_with if fea.shared_with is not None \
                else fea.name
            if isinstance(fea, SparseFeature):
                sparse_emb.append(
                    self.embed_dict[fea_key](x[fea_name].long())
                )
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                elif fea.pooling == 'none':
                    pooling_layer = IndityPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." %
                                     (["sum", "mean", "concat"], fea.pooling))
                fea_mask = InputMask()(x, fea)
                emb = pooling_layer(
                    self.embed_dict[fea_key](x[fea_name].long()), fea_mask)
                if fea.pooling != 'none':
                    # expand 1 dim for multiple feature concat
                    emb = emb.unsqueeze(1)
                sparse_emb.append(emb)
            else:
                dense_values.append(
                    x[fea_name].float()
                    .unsqueeze(1)
                )
        # ----- Combined Dense & Sparse Feature -----
        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
            if self.mlp:
                dense_values = self.mlp(dense_values)

        if len(sparse_emb) > 0:
            sparse_exists = True

        # Fusing Dense & Sparse embedding
        if squeeze_dim:
            if dense_exists and not sparse_exists:  # only dense features
                return dense_values

            elif not dense_exists and sparse_exists:  # only sparse feature
                sparse_emb = torch.cat(sparse_emb, dim=1)
                return sparse_emb

            elif dense_exists and sparse_exists:  # both sparse & dense feature
                sparse_emb = torch.cat(sparse_emb, dim=1)
                # concat dense value with sparse embedding
                return torch.cat([sparse_emb, dense_values], dim=1)
            else:
                raise ValueError("The input features can note be empty")

        else:  # return list of embedding
            if dense_exists and not sparse_exists:  # only dense features

                return dense_values

            elif not dense_exists and sparse_exists:  # only sparse feature

                return sparse_emb, dense_values  # list of sparse embedding

            elif dense_exists and sparse_exists:  # both sparse & dense feature
                return sparse_emb, dense_values
            else:
                raise ValueError("If keep the original shape: [batch_size, num_features, embed_dim], expected %s in feature list, got %s" % (
                    "SparseFeatures", features))


class LinearLayer(nn.Module):
    """Linear Module. It is the one linear
    transformation for input feature.

    Args:
        input_dim(int): input size of Linear module.

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)`
    """

    def __init__(self, input_dim, num_class=1, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, num_class, bias=True)

    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class MLP(nn.Module):
    def __init__(self, input_dim,
                 output_layer=False,
                 num_labels=1,
                 dims=None,
                 dropout=0,
                 bias=True,
                 activation="relu"):

        super().__init__()
        if dims is None:
            dims = []

        layers = []
        for i_dim in dims:
            l = nn.Linear(input_dim, i_dim, bias=bias)
            l.weight.data = RandomInitializer(i_dim, input_dim)

            layers.append(l)
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim

        if output_layer:
            layers.append(nn.Linear(input_dim, num_labels, bias=bias))
            layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CrossNetwork(nn.Module):
    """CrossNetwork  mentioned in the DCN paper.

    Args:
        input_dim(int): input dim of input tensor

    Shape:
        - Input: `(batch_size, *)`
        - Output: `(batch_size, *)`

    """

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        """
        : param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x = x0*self.w[i](x) + self.b[i] + x
        return x


# Pooling
class SumPooling(nn.Module):
    """Pooling the sequence embedding matrix by `sum`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)


class ConcatPooling(nn.Module):
    """Keep the origin sequence embedding shape

    Shape:
    - Input: `(batch_size, seq_length, embed_dim)`
    - Output: `(batch_size, seq_length, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class AveragePooling(nn.Module):
    """Pooling the sequence embedding matrix by `mean`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class IndityPooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return x
        else:
            return x


class InputMask(nn.Module):
    """
    Return inputs mask from given features

    Shape:
        - Input:
            x(dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape: `(batch_size, seq_len)`, \
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features(list or SparseFeature or SequenceFeature): Note that the elements in features are either all instances of SparseFeature or all instances of SequenceFeature.
        - Output:
            - if input Sparse: `(batch_size, num_features)`
            - if input Sequence: `(batch_size, num_features_seq, seq_length)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        for fea in features:
            if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature):
                if fea.padding_idx != None:
                    fea_mask = x[fea.name].long() != fea.padding_idx
                else:
                    fea_mask = x[fea.name].long() != -1
                mask.append(fea_mask.unsqueeze(1).float())
            else:
                raise ValueError(
                    "Only SparseFeature or SequenceFeature support to get mask.")
        return torch.cat(mask, dim=1)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(emb_dim, n_head)
        self.ln_1 = LayerNorm(emb_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(emb_dim, emb_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(emb_dim * 4, emb_dim))
        ]))
        self.ln_2 = LayerNorm(emb_dim)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# --------------------------------------------------------------


if __name__ == "__main__":
    from features import *
    feas = [DenseFeature(), SparseFeature(4)]
    test_emb = EmbeddingLayer(feas)

    x = torch.tensor([[1, 2, 3],
                      [0, 1, 2]])
    out_emb = test_emb(x)
    print(out_emb)

    """
    [tensor([1, 2, 3]), tensor([[0.0758, -0.0260, -0.3404, -0.4076,  0.7882, -0.7857],
        [1.5918,  2.0925, -0.5848, -0.3661, -0.5937,  0.9394],
        [0.7548,  0.1428, -1.1263,  0.3068,  1.3609,  0.3945]],
       grad_fn= < EmbeddingBackward0 >)]
    """
