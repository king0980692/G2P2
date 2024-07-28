import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *

class TRM(nn.Module):
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def __init__(self, features, mlp_params=None, **kwargs):
    # def __init__(self, emb_dim: int, n_layers: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        # self.token_embedding = nn.Embedding(args.vocab_size, args.transformer_width)
        embed_dim = features[0].embed_dim
        self.context_length = kwargs['context_length']
        self.features = features

        self.embedding = EmbeddingLayer(features)

        self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length,
                            embed_dim))                 

        self.ln_final = LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(
                torch.empty(embed_dim, 128))# 128: project to another size
                 
        self.transformer = Transformer(
				width=embed_dim,
				layers=2,
				heads=4,
				attn_mask=self.build_attention_mask())

    def forward(self, x_dict):

        x = self.embedding(x_dict, self.features, squeeze_dim=True)


        # x = self.token_embedding(text).type(self.dtype)  
        # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2) # NLD -> LND,
                               # batch_size * context_length * emb_dim 
                               # -> context_length * batch_size * emb_dim
        x = self.transformer(x)

        x = x.permute(1, 0, 2) #NLD -> LND,
                               # batch_size * context_length * emb_dim 
                               # -> context_length * batch_size * emb_dim

        x = self.ln_final(x) # [ batch_size, n_ctx, transformer.width]

        # take features from the eot （end of token）embedding 
        # (eot_token is the highest number in each sequence)
        # so there is node need to shorten the context length
        x = x[torch.arange(x.shape[0]), x_dict['u@s'].argmax(dim=-1)]  #
        # x = x @ self.text_projection

        return x


