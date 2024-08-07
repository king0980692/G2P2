from tqdm import trange, tqdm
import json
from multitask_amazon import multitask_data_generator
from sklearn.metrics import accuracy_score, f1_score
from data_graph import DataHelper
from model import CLIP, tokenize
import time
import math
import random
from random import sample
import torch
import argparse
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader
from collections import defaultdict
import os.path as osp
import sys

sys.path.append("../")

# from torch_geometric.loader import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)
# device = torch.device("cpu")
FType = torch.FloatTensor
LType = torch.LongTensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(seed)
    model = CLIP(args).to(device)

    # model.load_state_dict(torch.load('../G2P2_datasets/pretrain_model/Musical_Instruments/node_ttgt_8_12_10.pkl'))
    # model.load_state_dict(torch.load('./res/{}/node_ttgt_8&12_10_10.pkl'.format(data_name)))
    # model.load_state_dict(torch.load('./res/{}/node_ttgt_8&12_10.pkl'.format(data_name)))
    # model.load_state_dict(torch.load('./res/{}/new2_node_ttgt_8&12_10_4.pkl'.format(data_name)))
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    else:
        model.load_state_dict(torch.load(args.model))
    data_abbr = {
        "Musical_Instruments": "MI",
        "All_Beauty": "BE",
        "Industrial_and_Scientific": "IS",
        "Sports_and_Outdoors": "SO",
        "Toys_and_Games": "TG",
        "Arts_Crafts_and_Sewing": "AC",
        "reviews_Beauty_5": "BE5",
    }
    data_name = data_abbr.get(args.model.split("/")[1], "MI")

    print("Use model: ", args.model)

    user_id_set = set()
    item_id_set = set()

    file = f"tmp/{data_name}.{args.type}.u"
    with open(file) as f:
        for line in f:
            uid, iid, *_ = line.rstrip().split("\t")
            user_id_set.add(int(uid))
            item_id_set.add(int(iid))

    middle_idx = len(edge_index[0, :]) // 2
    # all_node_idx = list(range(torch.max(edge_index).cpu().item()+1)) #0-index
    all_node_idx = np.unique(arr_edge_index[:, :middle_idx])

    Data = DataHelper(arr_edge_index, args, all_node_idx)

    # ---------------------------------------------------
    print("check id dict is ok ...")
    loader = DataLoader(Data, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    for i_batch, sample_batched in tqdm(enumerate(loader)):
        s_n, t_n = sample_batched["s_n"], sample_batched["t_n"]
        s_n_arr = s_n.numpy()  # .reshape((1, -1))
        # t_n_arr = t_n.numpy().reshape(-1)
        s_n_text = [new_dict[i] for i in s_n_arr]

    # ---------------------------------------------------

    loader = DataLoader(Data, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    item_id_arr = np.array(list(item_id_set))
    user_id_arr = np.array(list(user_id_set))

    node_feas = []
    text_feas = []
    gt_feas = []
    tg_feas = []

    max_warm_id = len(node_f) - 1
    warm_mask = edge_index < max_warm_id
    combined_mask = warm_mask[0] & warm_mask[1]

    warm_edge_index = edge_index[:, combined_mask]

    for i_batch, sample_batched in enumerate(tqdm(loader)):
        s_n = sample_batched["s_n"].numpy()

        item_mask = np.isin(s_n, item_id_arr)

        raw_s_n_text = [
            item_prompt + " " + new_dict[i] + "."
            if is_item
            else user_prompt + new_dict[i]
            for i, is_item in zip(s_n, item_mask)
        ]

        # if True in item_mask:
        # import IPython;IPython.embed(colors='linux');exit(1)

        s_n_text = tokenize(
            raw_s_n_text, context_length=args.context_length).to(device)

        with torch.no_grad():
            if args.type == "val":
                if s_n.max() > max_warm_id:
                    alt_s_n = s_n[s_n >= max_warm_id]
                    warm_s_n = s_n[s_n < max_warm_id]
                    alt_s_n_text = [
                        item_prompt + new_dict[i]
                        if is_item
                        else user_prompt + new_dict[i]
                        for i, is_item in zip(alt_s_n, item_mask)
                    ]
                    alt_s_n_text = tokenize(
                        alt_s_n_text, context_length=args.context_length
                    ).to(device)

                    node_fea = model.encode_image(
                        warm_s_n, node_f, warm_edge_index)
                    alt_node_fea = model.encode_text(alt_s_n_text)
                    node_feas.append(
                        torch.cat([node_fea, alt_node_fea], dim=0))

                else:
                    node_fea = model.encode_image(s_n, node_f, warm_edge_index)
                    node_feas.append(node_fea)

            # otherwise using text embedding
            text_fea = model.encode_text(s_n_text)
            text_feas.append(text_fea)

    if args.type == "val":
        node_feas = torch.cat(node_feas, dim=0)

        node_output = []
        for i in trange(len(node_feas)):
            emb = list(map(lambda x: str(x), node_feas[i].tolist()))
            out = str(all_node_idx[i]) + "\t" + " ".join(emb)
            node_output.append(out)

        print(f"Save into tmp/{data_name}_g2p2_g_{args.type}.emb")
        with open(f"tmp/{data_name}_g2p2_g_{args.type}.emb", "w") as f:
            f.write("\n".join(node_output))

    text_feas = torch.cat(text_feas, dim=0)
    text_output = []
    for i in trange(len(text_feas)):
        emb = list(map(lambda x: str(x), text_feas[i].tolist()))
        out = str(all_node_idx[i]) + "\t" + " ".join(emb)
        text_output.append(out)

    print(f"Save into tmp/{data_name}_g2p2_t_{args.type}.emb")
    with open(f"tmp/{data_name}_g2p2_t_{args.type}.emb", "w") as f:
        f.write("\n".join(text_output))

    # Split embedding into user, item
    item_text_output = []
    user_text_output = []
    for i in trange(len(text_feas)):
        if all_node_idx[i] in item_id_arr:
            item_text_output.append(text_output[i])
        else:
            user_text_output.append(text_output[i])

    print(f"Save into tmp/{data_name}_g2p2_t_item_{args.type}.emb")
    with open(f"tmp/{data_name}_g2p2_t_item_{args.type}.emb", "w") as f:
        f.write("\n".join(item_text_output))

    print(f"Save into tmp/{data_name}_g2p2_t_user_{args.type}.emb")
    with open(f"tmp/{data_name}_g2p2_t_user_{args.type}.emb", "w") as f:
        f.write("\n".join(user_text_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--aggregation_times", type=int, default=2, help="Aggregation times"
    )
    parser.add_argument(
        "--hidden", type=str, default=16, help="number of hidden neurons"
    )
    parser.add_argument("--epoch_num", type=int,
                        default=101, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--neigh_num", type=int, default=3)

    parser.add_argument("--gnn_input", type=int, default=128)
    # parser.add_argument('--gnn_input', type=int, default=300)
    parser.add_argument("--gnn_hid", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)
    parser.add_argument("--edge_coef", type=float, default=0.1)

    parser.add_argument("--k_spt", type=int, default=2)
    parser.add_argument("--k_val", type=int, default=1)
    parser.add_argument("--k_qry", type=int, default=50)
    parser.add_argument("--n_way", type=int, default=5)

    parser.add_argument("--context_length", type=int, default=128)  # 120

    parser.add_argument("--data_name", type=str, default="Musical_Instruments")
    parser.add_argument(
        "--type", type=str, default="test", choices=["train", "val", "test", "support"]
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument(
        "--vocab_size", type=int, default=49408
    )  # decided by the given vocab

    # 2 heads, 6 layers, the first attempt
    # 8 heads, 12 layers, the second attempt

    args = parser.parse_args()
    data_name = args.data_name

    # check model path is exist
    if not osp.exists(args.model):
        print(f"model {args.model} not exist")
        sys.exit(1)

    print(
        "Load edge index from : ", "./tmp/{}_{}_edge.npy".format(
            data_name, args.type)
    )
    edge_index = np.load("./tmp/{}_{}_edge.npy".format(data_name, args.type))

    arr_edge_index = edge_index

    edge_index = torch.from_numpy(edge_index).to(device)

    # node_f = np.load('../cora/node_f_title.npy').astype(np.float32)
    print("Load node feature from : ", "./tmp/{}_f_m.npy".format(data_name))
    node_f = np.load("./tmp/{}_f_m.npy".format(data_name))
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)

    text_from_dict = {
        "test": "support",
    }

    text_from = (
        args.type if args.type not in text_from_dict else text_from_dict[args.type]
    )
    print("Load id->text from : ", text_from)
    tit_dict = json.load(
        open("./tmp/{}_{}_text.json".format(args.data_name, text_from))
    )

    new_dict = {}
    for key, text in tit_dict.items():
        new_dict[int(key)] = text

    # the_list = ['an arts crafts or sewing of', 'arts crafts or sewing of', 'arts crafts of', 'sewing of', 'art of']
    # the_list = ['followed by a vivid description of product is ']
    # item_prefix = [" ".join(["X"] * 4)]
    item_prefix = [""]
    item_prompt = [""] if args.type != "support" else item_prefix

    start = time.perf_counter()
    for Bob in range(len(item_prompt)):
        item_prompt = item_prompt[Bob]
        # user_prompt = "This user says:"
        user_prompt = ""
        # print('Prompt:', prompt_text)
        seed = int(math.pow(2, 0))
        main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
