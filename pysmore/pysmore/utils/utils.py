from typing import List
import os
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, FunctionTransformer
import itertools
import json
import importlib
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform
import scipy.sparse as sp

try:
    from pysmore.input.features import SparseFeature, DenseFeature, SequenceFeature
except:
    from input.features import SparseFeature, DenseFeature, SequenceFeature


def dynamic_import_cls(module_name, class_name):
    return getattr(importlib.import_module(name=module_name), class_name)


def _get_density(inter_df, n_users, n_items):
    return inter_df.shape[0] / (n_users * n_items)


def LoadPretrain(logger, args, feat_dims_size, lb_enc_list, col_names):
    pretrain_embs = [None] * len(col_names)

    for id, pretrain_file in enumerate(args.pretrain):
        if pretrain_file == "x":  # place holder
            continue
        enc = lb_enc_list[id]
        logger.info("Load Embedding of {} ".format(pretrain_file))
        embed = {}
        embedding_matrix = np.zeros((feat_dims_size[id], args.embed_dim))
        with open(pretrain_file) as f:
            for line in f.readlines():
                line = line.rstrip().split("\t")
                try:
                    node_id = enc.transform([line[0]])[0]
                except:
                    raise ValueError(
                        'Can\'t map node "{}" in :  embedding file {} '.format(
                            line[0], pretrain_file
                        )
                    )
                embed = np.fromstring(line[1], sep=" ")
                if len(embed) != args.embed_dim:
                    raise ValueError(
                        "The embedding size of pretrain {} "
                        "is {},not consistent with input: {} ".format(
                            pretrain_file, len(embed), args.embed_dim
                        )
                    )
                embedding_matrix[node_id] = embed

        # finish reading file
        pretrain_embs[id] = torch.from_numpy(embedding_matrix)

    return pretrain_embs


def save_embedding(
    user_embedding, item_embedding, rv_user_mapping, rv_item_mapping, saved_path
):
    print(f"\nSaving Embedding to {saved_path}")
    # emb = self.embedding.weight.detach()

    output = []
    # for _i in range(0, self.kwargs["user_num"]):
    for _i in trange(0, user_embedding.shape[0]):
        u_id = str(rv_user_mapping[_i])
        u_vec = user_embedding[_i].tolist()
        vec_str = " ".join([str(_v) for _v in u_vec])

        output.append(u_id + "\t" + vec_str + "\n")

    # for _i in range(0, self.kwargs["item_num"]):
    for _i in trange(0, item_embedding.shape[0]):
        i_id = str(rv_item_mapping[_i])
        i_vec = item_embedding[_i].tolist()
        vec_str = " ".join([str(_v) for _v in i_vec])

        output.append(i_id + "\t" + vec_str + "\n")

    with open(saved_path, "w") as f:
        f.writelines(output)


def generateNegSamples(n_sam, pop, length, sample_alpha):
    n_items = len(pop)
    if sample_alpha:
        sample = np.searchsorted(pop, np.random.rand(n_sam * length))
    else:
        sample = np.random.choice(n_items, size=n_sam * length)
    if length > 1:
        sample = sample.reshape((length, n_sam))
    return sample


def create_popularity(df, item_col, n_neg, cached_size, sample_alpha=0.0):
    pop = df.groupby(item_col).size()
    pop = pop.values**sample_alpha
    pop = pop.cumsum() / pop.sum()
    pop[-1] = 1

    neg_sam = None
    if cached_size:
        generated_length = int(cached_size // n_neg)
        if generated_length <= 1:
            sample_cache = 0
            print("No example store was used")
        else:
            neg_sam = generateNegSamples(n_neg, pop, generated_length, sample_alpha)
    else:
        print("No example store was used")
    return generated_length, pop, neg_sam


def create_sess_base_order(df, user_col, time_col, num_sess, timeSort=False):
    """
    Creating arrays to arrange data by time or not
    """
    if timeSort:
        baseOrder = np.argsort(df.groupby(user_col)[time_col].min().values)
    else:
        baseOrder = np.arange(num_sess - 1)
    return baseOrder


def create_offset_session_idx(df, user_col, time_col):
    df.sort_values([user_col, time_col], inplace=True)
    offsetSessions = np.zeros(df[user_col].nunique() + 1, dtype=np.int32)
    offsetSessions[1:] = df.groupby(user_col).size().cumsum()
    return offsetSessions


def create_sp_adj_mat(user_idx, item_idx, n_users, n_items):
    # Basic sparse matrix
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    R[user_idx, item_idx] = 1.0

    # Symmetric adjacency matrix
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    return R, adj_mat


def create_lil_graph_by_interaction_df(df, user_col, item_col):
    user_size = df[user_col].squeeze().max() + 1
    item_size = df[item_col].squeeze().max() + 1

    element_list = [list() for u in range(user_size)]

    for row in df.itertuples(index=False):
        element_list[row[0]].append((row[1], row[2]))

    rate_list = [list(map(lambda x: float(x[1]), l)) for l in element_list]
    user_list = [list(map(lambda x: x[0], l)) for l in element_list]

    return user_list, rate_list, user_size, item_size


def ParseFeature(
    input_names,
    input_types,
    sp_vocab_size,
    neg_sample=False,
    embed_dim=None,
    pooling_opt: str = "mean",
    ignore_feat=["y", "r", "t"],
):
    """

    This function parse every user and item feature.

    Return
    ------
    user_features : list
        List of SparseFeature or SequenceFeature
    item_features : list
        List of SparseFeature or SequenceFeature
    user_meta_cols : list
        List of meta field names of user features
    item_meta_cols : list
        List of meta field names of item features

    """

    if pooling_opt not in ["mean", "sum", "concat", "max", "none"]:
        raise ValueError("Invalid pooling method : {}".format(pooling_opt))

    user_features = []
    item_features = []

    user_meta_cols = []
    item_meta_cols = []

    idx = 0
    ignore_feat = ["y", "r", "t"]

    for feat_name, feat_type in zip(input_names, input_types):
        if feat_name in ignore_feat:
            continue
        elif "s" in feat_name:  # sequence feature
            if "u" in feat_type:
                user_meta_cols.append(feat_name)
                user_features.append(
                    SequenceFeature(
                        vocab_size=sp_vocab_size[idx],
                        name=feat_name,
                        pooling=pooling_opt,
                        embed_dim=embed_dim,
                    )
                )
            elif "i" in feat_type:
                item_meta_cols.append(feat_name)
                item_features.append(
                    SequenceFeature(
                        vocab_size=sp_vocab_size[idx],
                        name=feat_name,
                        pooling=pooling_opt,
                        embed_dim=embed_dim,
                    )
                )
            else:
                raise ValueError("Invalid feature name : {}".format(feat_name))
        elif "d" in feat_name:  # dense feature
            if "u" in feat_type:
                user_meta_cols.append(feat_name)
                user_features.append(DenseFeature(name=feat_name))
            elif "i" in feat_type:
                item_meta_cols.append(feat_name)
                item_features.append(DenseFeature(name=feat_name))
                if neg_sample:
                    item_features.append(DenseFeature(name=f"neg-{feat_name}"))
            else:
                raise ValueError("Invalid feature name : {}".format(feat_name))
        elif "c" in feat_name or "u" in feat_name or "i" in feat_name:
            if "u" in feat_type:
                if feat_name != "u":
                    user_meta_cols.append(feat_name)

                user_features.append(
                    SparseFeature(
                        vocab_size=sp_vocab_size[idx],
                        name=feat_name,
                        embed_dim=embed_dim,
                    )
                )
                idx += 1
            elif "i" in feat_type:
                if feat_name != "i":
                    item_meta_cols.append(feat_name)
                item_features.append(
                    SparseFeature(
                        vocab_size=sp_vocab_size[idx],
                        name=feat_name,
                        embed_dim=embed_dim,
                    )
                )
                if neg_sample:
                    item_features.append(
                        SparseFeature(
                            vocab_size=sp_vocab_size[idx],
                            name=f"neg-{feat_name}",
                            shared_with=f"{feat_name}",
                            embed_dim=embed_dim,
                        )
                    )
                idx += 1
        else:
            # if feat_name[0] == 'F':
            raise ValueError("Invalid feature name : {}".format(feat_name))

    return (user_features, item_features, user_meta_cols, item_meta_cols)


def _check_input_format(input_format, file_name, seperator):
    with open(file_name) as f:
        line = f.readline().rstrip().split(seperator)

        if len(line) != len(eval(input_format)):
            raise ValueError(
                'The number of fields in the input format {} is not consistent with "{}" format {}'.format(
                    input_format, file_name, line
                )
            )


def ParseInput(input_format, task="rank"):
    """
    This function do the following things:
        1. evaluates the input format string
        2. picks the field name by some heuristic
        3. if the task is not retrieval, pad the field with 'u'
        4. give every field a unique name and type
    """

    flatten_format = eval(input_format)

    # Eval the '-' symbol into range
    parsed_format = []
    names_set = {}
    for field in flatten_format:
        if "-" in field:
            _type = field[0]
            f_s, f_e = field.split("-")
            f_s, f_e = int(f_s[1:]), int(f_e[1:])
            parsed_format += [f"{_type}{i}" for i in range(f_s, f_e + 1)]
        else:
            if field not in names_set:
                names_set[field] = 1
                parsed_format.append(field)
            else:
                names_set[field] += 1
                parsed_format.append(f"{field}{names_set[field]}")

    # Get field name by guessing some possible names
    user_field_name = parse_fields(parsed_format, ["u", "user", "users"])
    item_field_name = parse_fields(parsed_format, ["i", "item", "items"])
    target_field_name = parse_fields(parsed_format, ["r", "y", "rating", "target"])
    time_field_name = parse_fields(parsed_format, ["t", "time"])

    ignore_feat = [time_field_name, target_field_name]
    # Pad the field with 'u' if task is not retrieval
    if task == "rank":
        parsed_format = [
            f"F@{format}" if format not in ignore_feat else format
            for format in parsed_format
        ]
        user_field_name = "F@" + user_field_name
        item_field_name = "F@" + item_field_name

    col_names = parsed_format

    # col_types
    col_types = []
    for field in parsed_format:
        if "d" in field:
            col_types.append(f"{field[0]}.d")
        elif "s" in field:
            col_types.append(f"{field[0]}.s")
        elif target_field_name in field:
            col_types.append("y")
        elif "c" in field:
            col_types.append(f"{field[0]}.c")
        elif user_field_name in field:
            col_types.append("u.c")
        elif item_field_name in field:
            col_types.append("i.c")
        elif time_field_name in field:
            col_types.append("t")
        else:
            raise ValueError("Invalid format : {}".format(field))

    # Parse field with its type
    sparse_fields = [
        field
        for field in parsed_format
        if "c" in field.split("@")[-1] or "u" in field or "i" in field
    ]
    dense_fields = [field for field in parsed_format if "d" in field.split("@")[-1]]
    sequence_fields = [field for field in parsed_format if "s" in field.split("@")[-1]]

    return (
        col_names,
        col_types,
        user_field_name,
        item_field_name,
        target_field_name,
        time_field_name,
        sparse_fields,
        dense_fields,
        sequence_fields,
    )


def FeaturePreprocessing(
    df,
    target_col,
    sparse_features,
    dense_features,
    seq_features,
    user_col,
    item_col,
    enc_policy="default",
    sp_vocab_size=None,
    enc_list=None,
    vocab_list=None,
    tokenizer=None,
):
    if sp_vocab_size is None:
        sp_vocab_size = []

    user_map = {}
    rv_user_map = {}
    item_map = {}
    rv_item_map = {}

    # Target Feature Processing
    n_classes = None
    if target_col != "":
        df[target_col] = (
            df[target_col].astype("float").astype("int")
        )  # otherwise label encoding
        if enc_list is None:
            n_classes = df[target_col].nunique()
            # lbe = LabelEncoder()
            # df[target_col] = lbe.fit_transform(df[target_col]) # 0-indexing
        else:  # for val set encoding
            n_classes = None

        # Sequence Feature Processing

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    enc_vocab_list = []

    for id, feature in enumerate(seq_features):
        if vocab_list:  # apply encoder to validation set
            vocab = vocab_list[id]
            vocab_transform = VocabTransform(vocab)
            df[feature] = df[feature].apply(
                lambda x: vocab_transform(x.lower().split())
            )

        else:
            """
                build vocab table
            """
            vocab = build_vocab_from_iterator(
                yield_tokens(df[feature].tolist()),
                specials=["<pad>", "<unk>"],
                special_first=True,
            )
            vocab.set_default_index(vocab["<pad>"])
            vocab.set_default_index(vocab["<unk>"])
            enc_vocab_list.append(vocab)

            vocab_transform = VocabTransform(vocab)
            sp_vocab_size.append(len(vocab))
            df[feature] = df[feature].apply(
                lambda x: vocab_transform(x.lower().split())
            )

        # Sparse Feature Processing
    df[sparse_features] = df[sparse_features].fillna("0")
    df[dense_features] = df[dense_features].fillna(0)

    lb_enc_list = []

    for id, feature in enumerate(sparse_features):
        # """
        # max_sparse_feat_dims.append(df[feature].nunique())  # n_unique

        if enc_list:  # apply encoder to validation set
            if enc_policy == "none":
                df[feature] = df[feature].astype("int")

                lbe = enc_list[id]

                new_classes = [
                    label for label in df[feature].unique() if label not in lbe.classes_
                ]
                if len(new_classes):
                    lbe.classes_ = np.unique(
                        np.concatenate([lbe.classes_, new_classes])
                    )
                    sp_vocab_size[id] = len(lbe.classes_)

            elif enc_policy == "handle_unk":
                lbe = enc_list[id]
                df[feature] = [x if x in lbe.classes_ else "<unk>" for x in df[feature]]
                df[feature] = lbe.transform(df[feature])
            else:  # defualt
                lbe = enc_list[id]

                old_class = lbe.classes_
                ori_ = lbe.transform(old_class)

                new_classes = [
                    label for label in df[feature].unique() if label not in lbe.classes_
                ]
                if len(new_classes):
                    # lbe.classes_ = np.unique(
                    #     np.concatenate([lbe.classes_, new_classes])
                    # )
                    lbe.classes_ = np.append(lbe.classes_, new_classes)

                    sp_vocab_size[id] = len(lbe.classes_)

                df[feature] = lbe.transform(df[feature])

                assert all(lbe.transform(old_class) == ori_)

                # lbe_dict = dict(zip(lbe.classes_, lbe.transform(lbe.classes_)))
                # df[feature] = df[feature].apply(
                #     lambda x: lbe_dict.get(x, np.nan))

                # if len(max_sparse_feat_dims):
                #     max_sparse_feat_dims[id] = len(lbe.classes_)

                # select the warm users & warm items
                # df = df[df[feature].notna()]
                # df = df.astype('int64')
        else:
            if enc_policy == "none":
                df[feature] = df[feature].astype("int")
                idempoten_map = {x: x for x in df[feature].unique()}

                lbe = LabelEncoder()
                lbe.classes_ = np.array(sorted(idempoten_map, key=idempoten_map.get))

                lb_enc_list.append(lbe)
                sp_vocab_size.append(1 + df[feature].max())

            elif enc_policy == "handle_unk":
                uniq_vocab = list(df[feature].unique())
                uniq_vocab.append("<unk>")
                lbe = LabelEncoder().fit(uniq_vocab)
                # OrdinalEncoder().fit(uniq_vocab)

                df[feature] = lbe.transform(df[feature])

                lb_enc_list.append(lbe)

                sp_vocab_size.append(len(uniq_vocab))  # n_unique
            else:
                lbe = LabelEncoder()
                df[feature] = lbe.fit_transform(df[feature])  # 0-indexing
                sp_vocab_size.append(len(lbe.classes_))  # n_unique
                lb_enc_list.append(lbe)

                if feature == user_col:
                    rv_user_map = {
                        encode_id: raw_id
                        for encode_id, raw_id in enumerate(lbe.classes_)
                    }
                    user_map = {
                        raw_id: encode_id
                        for encode_id, raw_id in enumerate(lbe.classes_)
                    }

                if feature == item_col:
                    rv_item_map = {
                        encode_id: raw_id
                        for encode_id, raw_id in enumerate(lbe.classes_)
                    }
                    item_map = {
                        raw_id: encode_id
                        for encode_id, raw_id in enumerate(lbe.classes_)
                    }
        # """

    # Dense Feature Processing
    for id, feature in enumerate(dense_features):
        df[dense_features] = df[dense_features].clip(
            lower=0
        )  # make sure all values are non-negative

        feat_trans = MinMaxScaler()  # scaler dense feature
        df[dense_features] = feat_trans.fit_transform(df[dense_features])

        # feat_trans = FunctionTransformer(np.log1p)
        # df[dense_features] = feat_trans.fit_transform(df[dense_features])

    return (
        df,
        n_classes,
        lb_enc_list,
        enc_vocab_list,
        user_map,
        item_map,
        rv_user_map,
        rv_item_map,
    )


def parse_fields(
    fields: List[str], possible_field_name: List[str], force: bool = False
) -> int:
    target_idx = -1

    for field_name in possible_field_name:
        try:
            target_idx = fields.index(field_name)
        except:
            pass

    if force:
        assert target_idx != -1, "Not found valid field name like : {}".format(
            possible_field_name
        )

    if target_idx == -1:
        return ""
    else:
        return fields[target_idx]


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    rv_column_dict = {i: x for i, x in enumerate(df[column_name].unique())}

    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype("int")

    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict, rv_column_dict


def create_user_list(
    df, user_size, user_field_name, item_field_name, target_field_name
):
    user_list = [list() for u in range(user_size)]

    for row in df.itertuples():
        # user_list[row[user_field_name]].append((row.time, row.item, row.rate))
        user_list[getattr(row, user_field_name)].append(
            (getattr(row, item_field_name), getattr(row, target_field_name))
        )
    return user_list


def split_train_valid(df, user_size, test_size=0.2, time_order=False):
    """Split a dataset into `train_user_list` and `test_user_list`.
    Because it needs `user_list` for splitting dataset as `time_order` is set,
    Returning `user_list` data structure will be a good choice."""
    # TODO: Handle duplicated items

    if not time_order:
        test_idx = np.random.choice(len(df), size=int(len(df) * test_size))
        train_idx = list(set(range(len(df))) - set(test_idx))
        test_df = df.loc[test_idx].reset_index(drop=True)
        train_df = df.loc[train_idx].reset_index(drop=True)

    else:
        raise NotImplementedError
        """
        total_user_list = create_user_list(df, user_size)
        train_user_list = [None] * len(total_user_list)
        test_user_list = [None] * len(total_user_list)
        for user, item_list in enumerate(total_user_list):
            # Choose latest item
            item_list = sorted(item_list, key=lambda x: x[0])
            # Split item
            test_item = item_list[math.ceil(len(item_list)*(1-test_size)):]
            train_item = item_list[:math.ceil(len(item_list)*(1-test_size))]
            # Register to each user list
            test_user_list[user] = test_item
            train_user_list[user] = train_item
        """

    return train_df, test_df


def create_user_edge_list(
    df, user_size, user_field_name, item_field_name, target_field_name
):
    user_list = create_user_list(
        df, user_size, user_field_name, item_field_name, target_field_name
    )

    rate_list = [list(map(lambda x: x[1], l)) for l in user_list]
    user_list = [list(map(lambda x: x[0], l)) for l in user_list]

    return user_list, rate_list


def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair


def print_status(
    args,
    model,
    sparse_exist,
    seq_exist,
    dense_exist,
):
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # -------------------
    print(f"{BLUE} === Task Specfic Setting === {ENDC}")
    ordered_args = [
        "task",
    ]
    hightlighted_args = {
        "task": f"{BOLD}{YELLOW}",
    }

    for arg in ordered_args:
        prefix_color = hightlighted_args.get(arg, "")
        print(
            f"{GREEN}\t{arg:<20} {ENDC} {prefix_color} {args.__getattribute__(arg)} {ENDC}"
        )

    # -------------------
    print(f"{BLUE} === Training Setting === {ENDC}")
    ordered_args = [
        "device",
        "model",
        "loss_fn",
        "optim",
        "lr",
        "weight_decay",
        "max_epochs",
        "log_interval",
        "es_patience",
        "metrics",
    ]
    hightlighted_args = {
        "device": f"{BOLD}{YELLOW}" if "cpu" in args.device else "",
        "model": f"{BOLD}{MAGENTA}",
        "loss_fn": f"{BOLD}{MAGENTA}",
        "lr": f"{BOLD}{YELLOW}",
        "optim": f"{BOLD}{YELLOW}",
        "metrics": f"{BOLD}{YELLOW}" if args.metrics is not None else "",
    }
    for arg in ordered_args:
        prefix_color = hightlighted_args.get(arg, "")
        print(
            f"{GREEN}\t{arg:<20} {ENDC} {prefix_color} {args.__getattribute__(arg)} {ENDC}"
        )

    # -------------------
    print(f"{BLUE} === Data Setting === {ENDC}")
    ordered_args = [
        "batch_size",
        "worker",
    ]
    for arg in ordered_args:
        prefix_color = hightlighted_args.get(arg, "")
        print(
            f"{GREEN}\t{arg:<20} {ENDC} {prefix_color} {args.__getattribute__(arg)} {ENDC}"
        )

    # -------------------

    print(f"{BLUE} === Feature Specific Setting === {ENDC}")
    ordered_args = ["num_neg"]
    hightlighted_args = {
        "num_neg": f"{BOLD}{YELLOW}" if args.task == "retrieval" else "",
    }
    print(f"{CYAN} [Sparse Feature] {ENDC}")
    for arg in ordered_args:
        prefix_color = hightlighted_args.get(arg, "")
        print(
            f"{GREEN}\t{arg:<20} {ENDC} {prefix_color} {args.__getattribute__(arg)} {ENDC}"
        )
    # if sparse_exist:
    ordered_args = ["embed_dim"]
    hightlighted_args = {
        "embed_dim": f"{BOLD}{YELLOW}" if args.embed_dim is not None else "",
    }

    print(f"{CYAN} [Sparse Feature] {ENDC}")
    for arg in ordered_args:
        prefix_color = hightlighted_args.get(arg, "")
        print(
            f"{GREEN}\t{arg:<20} {ENDC} {prefix_color} {args.__getattribute__(arg)} {ENDC}"
        )

    # if seq_exist:
    print(f"{CYAN} [Sequence Feature] {ENDC}")
    ordered_args = ["context_length", "pooling_opt"]
    hightlighted_args = {
        "pooling_opt": f"{BOLD}{YELLOW}",
    }

    for arg in ordered_args:
        prefix_color = hightlighted_args.get(arg, "")
        print(
            f"{GREEN}\t{arg:<20} {ENDC} {prefix_color} {args.__getattribute__(arg)} {ENDC}"
        )

    # -------------------

    print(f"{BLUE} === Output Setting === {ENDC}")
    ordered_args = ["saved_option", "saved_path"]
    hightlighted_args = {}

    for arg in ordered_args:
        prefix_color = hightlighted_args.get(arg, "")
        print(
            f"{GREEN}\t{arg:<20} {ENDC} {prefix_color} {args.__getattribute__(arg)} {ENDC}"
        )
    print(f"\n\n{BLUE} === Ready to Train === {ENDC}\n")

    import torchinfo

    torchinfo.summary(model)
