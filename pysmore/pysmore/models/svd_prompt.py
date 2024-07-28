from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.layers import *
    from input import *
    from models.mf import mf
    from models.simple_tokenizer import SimpleTokenizer as _Tokenizer
    import models.CLIP as model
except ModuleNotFoundError:
    from pysmore.models.layers import *
    from pysmore.input import *
    from pysmore.models.mf import mf
    from pysmore.models.simple_tokenizer import SimpleTokenizer as _Tokenizer
    import pysmore.models.CLIP as model


_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        self.transformer = self.transformer.to(prompts.device)
        self.ln_final = self.ln_final.to(prompts.device)
        text_projection = self.text_projection.to(prompts.device)
        positional_embedding = self.positional_embedding.to(prompts.device)

        x = prompts + positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, args, clip_model, n_items, item_texts, g_texts=None):
        super().__init__()
        self.vars = nn.ParameterList()
        n_cls = n_items

        # n_cls = len(classnames)
        n_ctx = args.coop_n_ctx
        # n_ctx = 4

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        if args.ctx_init and False:
            # use given words to initialize context vectors
            if args.class_specific:
                ctx_vectors = []
                for ctx_list in g_texts:
                    prompt = model.tokenize(
                        ctx_list, context_length=args.context_length)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(
                            prompt).type(dtype)
                    ctx_vector = embedding[:, 1: 1 + n_ctx, :]
                    ctx_vector = torch.mean(ctx_vector, dim=0)
                    ctx_vectors.append(ctx_vector)
                ctx_vectors = torch.stack(ctx_vectors)
            else:
                ##############
                #### here ####
                ##############
                temp = []
                for ctx_list in g_texts:
                    temp += ctx_list
                prompt = model.tokenize(
                    temp, context_length=args.context_length)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vector = embedding[:, 1: 1 + n_ctx, :]
                ctx_vectors = torch.mean(ctx_vector, dim=0)
            # print('ctx_vectors.shape', ctx_vectors.shape)
        else:
            if args.class_specific:
                # print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ##############
                #### here ####
                ##############
                # print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            # nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.xavier_normal_(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        # item_texts = list(item_texts.values())

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.vars.append(self.ctx)

        item_texts = [name.replace("_", " ") for name in item_texts]
        name_lens = [len(_tokenizer.encode(name)) for name in item_texts]
        prompts = [prompt_prefix + " " + name + "." for name in item_texts]

        tmp = []
        for p in prompts:
            tmp.append(model.tokenize(p, context_length=args.context_length))
        tokenized_prompts = torch.cat(tmp)

        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def parameters(self):
        return self.vars


class svd_prompt(torch.nn.Module):
    def __init__(self, features, mlp_params=None, **kwargs):
        super(svd_prompt, self).__init__()

        self.features = features

        self.n_dims = sum([fea.embed_dim for fea in features])

        self.user_dim = -1
        self.num_users = -1
        self.item_dim = -1
        self.num_items = -1
        for fea in features:
            if 'i' in fea.name:
                self.item_dim = fea.embed_dim
                self.num_items = fea.vocab_size
            elif 'u' in fea.name:
                self.user_dim = fea.embed_dim
                self.num_users = fea.vocab_size

        self.model = mf(features, mlp_params=mlp_params, **kwargs)

        self.bias = nn.ParameterDict()
        for fea in features:
            # torch.manual_seed(0)
            self.bias[fea.name] = nn.Parameter(torch.randn(fea.vocab_size))
            # self.bias[fea.name] = nn.Parameter(torch.zeros(fea.vocab_size))

        self.bias['global_bias'] = nn.Parameter(torch.zeros(1))

        # self.user_embedding = kwargs['pretrain_embs'][0].to(kwargs['device'])
        # self.user_embedding.requires_grad_(False)

        self.item_embedding = kwargs['pretrain_embs'][1].to(kwargs['device'])
        self.item_embedding.requires_grad_(False)
        self.item_embedding = self.item_embedding.to(torch.float32)

        self.text_encoder = TextEncoder(kwargs['clip_model'])

        # self.item_prompt = nn.parameter.Parameter(torch.randn(self.num_items, self.item_dim))
        self.prompt_learner = PromptLearner(kwargs['clip_cfg'],
                                            kwargs['clip_model'],
                                            self.num_users,
                                            kwargs['id_texts'])

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(
            kwargs['device'])

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        for name, param in kwargs['clip_model'].named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        """
        user_emb, item_emb = self.get_embedding()
        tmp = torch.cat([
            self.prompt_learner.token_prefix,
            self.prompt_learner.ctx.unsqueeze(0).expand(
                self.num_items, -1, -1),
            self.prompt_learner.token_suffix
            ], dim=1)

        item_embedding = []
        batch = 128
        with torch.no_grad():
            for i in trange(0,self.num_items,batch):
                ctx_vec = tmp[i:i+batch]
                ctx_vec = ctx_vec.to(self.tokenized_prompts.device)
                desc_vec = self.tokenized_prompts[i:i+batch]
                item_embedding.append(self.text_encoder(ctx_vec, desc_vec))
        item_embedding = torch.cat(item_embedding, dim=0)
        """

    def get_embedding(self):
        print("Getting Embedding")
        item_embedding = self.item_embedding

        # item_embedding = torch.zeros((self.num_items, self.item_dim))
        user_embedding = []
        batch = 128
        with torch.no_grad():
            for i in trange(0, self.num_users, batch):
                ctx_vec = self.prompt_learner()[i:i+batch]
                ctx_vec = ctx_vec.to(self.tokenized_prompts.device)
                desc_vec = self.tokenized_prompts[i:i+batch]
                user_embedding.append(self.text_encoder(ctx_vec, desc_vec))

        user_embedding = torch.cat(user_embedding, dim=0)
        return user_embedding, item_embedding

    def get_embedding2(self):
        print("Getting Embedding")
        user_embedding = self.user_embedding
        # item_embedding = self.item_embedding
        # return user_embedding, item_embedding

        # item_embedding = torch.zeros((self.num_items, self.item_dim))
        item_embedding = []
        batch = 128
        with torch.no_grad():
            for i in trange(0, self.num_items, batch):
                ctx_vec = self.prompt_learner()[i:i+batch]
                ctx_vec = ctx_vec.to(self.tokenized_prompts.device)
                desc_vec = self.tokenized_prompts[i:i+batch]
                item_embedding.append(self.text_encoder(ctx_vec, desc_vec))

        item_embedding = torch.cat(item_embedding, dim=0)
        return user_embedding, item_embedding

    def forward(self, x, debug=False):

        forward_features = [fea for fea in self.features if fea.name in x]

        key = 'F@u'
        prompts = self.prompt_learner()[x[key]]
        tokenized_prompts = self.tokenized_prompts[x[key]]
        user_features = self.text_encoder(prompts, tokenized_prompts)

        key = 'F@i'
        item_features = self.item_embedding[x[key]]

        """
        item_features = self.text_encoder(prompts, tokenized_prompts)
        user_features = self.user_embedding[x['u']]
        """

        score = torch.mul(user_features, item_features).sum(dim=1)

        norm = torch.norm(user_features, dim=1) * \
            torch.norm(item_features, dim=1)

        norm += 1e-8  # avoid nan

        score = score / norm

        for fea in forward_features:
            score += self.bias[fea.name][x[fea.name]]

        score += self.bias['global_bias']
        score = torch.clamp(score, min=1, max=5)

        return score

        # return self.model(x)
