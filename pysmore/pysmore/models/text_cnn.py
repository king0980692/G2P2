import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *

class text_cnn(nn.Module):
    def __init__(self, features, mlp_params=None, **kwargs):
    # def __init__(self, config):
        super(text_cnn, self).__init__()

        self.dropout = kwargs['dropout']

        
        self.dims = features[0].embed_dim
        self.embedding = EmbeddingLayer(features)
        # self.convs = ConvLayer(input_dim=1,
                               # out_dims=[256, 256, 256],
                               # filter_sizes=[(k, self.dims) for k in [2,3,4]],
                               # dropout=mlp_params["dropout"],
                               # )

        self.features = features

        self.out_channel = 256
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, self.out_channel, (k, 300)) for k in [2,3,4]])

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(self.out_channel * len([2,3,4]), kwargs['n_classes'])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x_dict):
        out, _ = self.embedding(x_dict, self.features, squeeze_dim=False)
        out = out[0]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs2], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
