from tqdm import tqdm, trange
import os
import os.path as osp
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from data import DataHelper
from sklearn import preprocessing
import json


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
    Data = DataHelper(arr_edge_index, args)

    # check if dict is ok
    print("check id dict is ok")
    # loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # for i_batch, sample_batched in tqdm(enumerate(loader)):
    #     s_n, t_n = sample_batched["s_n"], sample_batched["t_n"]
    #     s_n_arr = s_n.numpy()  # .reshape((1, -1))
    #     t_n_arr = t_n.numpy().reshape(-1)
    #     s_n_text, t_n_text = (
    #         [new_dict[i] for i in s_n_arr],
    #         [new_dict[j] for j in t_n_arr],
    #     )

    model.train()

    iter_loss = []
    for j in range(args.epoch_num):
        loader = DataLoader(
            Data, batch_size=args.batch_size, shuffle=True, num_workers=10
        )
        for i_batch, sample_batched in tqdm(enumerate(loader), total=len(loader)):
            s_n, t_n = sample_batched["s_n"], sample_batched["t_n"]
            s_n_arr = s_n.numpy()  # .reshape((1, -1))
            t_n_arr = t_n.numpy().reshape(-1)
            s_n_text, t_n_text = (
                [new_dict[i] for i in s_n_arr],
                [new_dict[j] for j in t_n_arr],
            )
            s_n_text, t_n_text = (
                tokenize(s_n_text, context_length=args.context_length).to(device),
                tokenize(t_n_text, context_length=args.context_length).to(device),
            )

            s_n, t_n = s_n.type(LType).to(device), t_n.type(LType).to(device)
            loss = model.forward(
                node_f, edge_index, s_n, t_n, s_n_text, t_n_text, device
            )
            iter_loss.append(loss)
            if j == 0 and i_batch % 100 == 0:
                print("{}th loss in the first epoch:{}".format(i_batch, loss))
        try:
            if osp.exists("./res/{}".format(args.data_name)) == False:
                os.makedirs("./res/{}".format(args.data_name))
            torch.save(
                model.state_dict(),
                "./res/{}/new2_node_ttgt_8&12_10_{}.pkl".format(args.data_name, j),
            )
        except:
            import ipdb

            ipdb.set_trace()
        # break
        print("{}th epoch loss:{}".format(j + 1, loss))
    torch.save(
        model.state_dict(), "./res/{}/new2_node_ttgt_8&12_10.pkl".format(args.data_name)
    )
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--aggregation_times", type=int, default=2, help="Aggregation times"
    )
    parser.add_argument("--epoch_num", type=int, default=10, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--edge_coef", type=float, default=10)
    parser.add_argument("--neigh_num", type=int, default=3)

    parser.add_argument("--gnn_input", type=int, default=128)
    parser.add_argument("--gnn_hid", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)

    parser.add_argument("--context_length", type=int, default=128)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)  # 49408
    parser.add_argument("--data_name", type=str, default="Musical_Instruments")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    tit_list = []
    tit_dict = json.load(open("./tmp/{}_{}_text.json".format(args.data_name, "train")))
    new_dict = {}

    for i in range(len(tit_dict)):
        new_dict[i] = tit_dict[str(i)]

    edge_index = np.load("./tmp/{}_train_edge.npy".format(args.data_name))

    arr_edge_index = edge_index

    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load("./tmp/{}_f_m.npy".format(args.data_name))
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    # node_f = torch.from_numpy(node_f).to(device)

    node_f = torch.nn.init.xavier_uniform_(torch.empty(node_f.shape)).to(device)

    # len(edge_index[0,:].unique())
    assert len(edge_index[0, :].unique()) == len(new_dict)

    start = time.perf_counter()

    seed = 1
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
