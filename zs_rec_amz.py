import sys

sys.path.append('../')
import os.path as osp
from collections import defaultdict
# from torch_geometric.loader import DataLoader
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
from data_graph import DataHelper
from sklearn.metrics import accuracy_score, f1_score
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from multitask_amazon import multitask_data_generator
from sklearn import preprocessing
import json

from tqdm import trange, tqdm


data_name = 'Musical_Instruments'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('device', device)
# device = torch.device("cpu")
FType = torch.FloatTensor
LType = torch.LongTensor



# node_f = np.load('../cora/node_f_title.npy').astype(np.float32)
node_f = np.load('./tmp/{}_f_m.npy'.format(data_name))
node_f = preprocessing.StandardScaler().fit_transform(node_f)
node_f = torch.from_numpy(node_f).to(device)



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
    model.load_state_dict(torch.load('./res/{}/test_node_ttgt_8&12_10_2.pkl'.format(data_name)))

    name_list = ["train", "test"]
    user_id_set = set()
    item_id_set = set()
    # for file in ["tmp/MI.train.u", "tmp/MI.support.u", "tmp/MI.test.u"]:
    
    for f_i, file in enumerate(["tmp/MI.test.u","tmp/MI.train.u" ]):

        src_name = "train" if f_i == 1 else "support"

        ##############
        # Load Graph #
        ##############
        edge_index = np.load('./tmp/{}_{}_edge.npy'.format(data_name,src_name))

        arr_edge_index = edge_index

        edge_index = torch.from_numpy(edge_index).to(device)

        ##############
        # Load Text  #
        ##############
        tit_dict = json.load(open('./tmp/{}_{}_text.json'.format(args.data_name, src_name)))
        new_dict = {}
        for key, text in tit_dict.items():
            new_dict[int(key)] = text
            

        with open(file) as f:
            for line in f :
                uid, iid, _ = line.rstrip().split('\t')
                user_id_set.add(int(uid))
                item_id_set.add(int(iid))

        middle_idx = len(edge_index[0,:])//2
        # all_node_idx = list(range(torch.max(edge_index).cpu().item()+1)) #0-index
        all_node_idx = np.unique(arr_edge_index[:,:middle_idx]) # One-direction graph

        Data = DataHelper(arr_edge_index, args, all_node_idx)
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)

        item_id_arr = np.array(list(item_id_set))
        user_id_arr = np.array(list(user_id_set))

        node_feas = []
        u_node_feas = []
        i_node_feas = []

        text_feas = []
        u_text_feas = []
        i_text_feas = []

        for i_batch, sample_batched in enumerate(tqdm(loader)):
            s_n = sample_batched['s_n'].numpy()

            item_mask = np.isin(s_n, item_id_arr)

            s_n_text = [ item_prompt+new_dict[i] if is_item 
                        else user_prompt+new_dict[i]\
                        for i, is_item in zip(s_n, item_mask) ] 
            s_n_text = tokenize(s_n_text, context_length=args.context_length).to(device)

            with torch.no_grad():
                if f_i == 1:
                    node_fea = model.encode_image(s_n, node_f, edge_index) # s_n
                    node_feas.append(node_fea)
                    u_node_feas.append(node_fea[~item_mask])
                    i_node_feas.append(node_fea[item_mask])

                text_fea = model.encode_text(s_n_text) # s_n_text
                text_feas.append(text_fea)
                u_text_feas.append(text_fea[~item_mask])
                i_text_feas.append(text_fea[item_mask])


        save_type = ['test', 'train']

        if f_i == 1:
            node_feas = torch.cat(node_feas, dim=0)
            u_node_feas = torch.cat(u_node_feas, dim=0)
            i_node_feas = torch.cat(i_node_feas, dim=0)
            iter_list = zip(["n", "t"],
                            [node_fea, u_node_feas, u_text_feas], 
                            [node_fea, i_node_feas, i_text_feas])

            ## Save Node Embedding
            output = []
            for i in trange(len(node_feas)):
                emb = list(map(lambda x: str(x), node_feas[i].tolist()))
                out = str(all_node_idx[i]) + "\t" + " ".join(emb)
                output.append(out)

            with open(f"tmp/g2p2.{save_type[f_i]}.emb", 'w') as f:
                f.write('\n'.join(output))

        elif f_i == 0:
            text_feas = torch.cat(text_feas, dim=0)
            u_text_feas = torch.cat(u_text_feas, dim=0)
            i_text_feas = torch.cat(i_text_feas, dim=0)
            iter_list = zip(["t"],
                            [u_text_feas],
                            [i_text_feas])

        for _type, u_emb, i_emb in iter_list:

            output = []
            for i in trange(len(u_emb)):
                emb = list(map(lambda x: str(x), u_emb[i].tolist()))
                out = str(all_node_idx[i]) + "\t" + " ".join(emb)
                output.append(out)

            with open(f"tmp/g2p2_{_type}.{save_type[f_i]}.u.emb", 'w') as f:
                f.write('\n'.join(output))

            output = []
            for i in trange(len(i_emb)):
                emb = list(map(lambda x: str(x), i_emb[i].tolist()))
                out = str(all_node_idx[i]) + "\t" + " ".join(emb)
                output.append(out)

            with open(f"tmp/g2p2_{_type}.{save_type[f_i]}.i.emb", 'w') as f:
                f.write('\n'.join(output))


        output = []
        for i in trange(len(text_feas)):
            emb = list(map(lambda x: str(x), text_feas[i].tolist()))
            out = str(all_node_idx[i]) + "\t" + " ".join(emb)
            output.append(out)

        with open(f"tmp/g2p2.{save_type[f_i]}.emb", 'w') as f:
            f.write('\n'.join(output))


    exit(1)
    ## --------------------------------------
    user_feas = []
    item_feas = []
    
    output_node_emb = np.zeros((all_node_idx[-1]+1, args.embed_dim))
    print("Get user embedding ...")
    for i_batch in trange(0, len(user_id_set), args.batch_size):
        batch_nodes = user_id_arr[i_batch:i_batch+args.batch_size]
        
        # batch_texts = [new_dict[i] for i in batch_nodes]
        # batch_texts = tokenize(batch_texts, context_length=args.context_length).to(device)

        with torch.no_grad():
            user_fea = model.encode_image(batch_nodes, node_f, edge_index)
            # text_fea = model.encode_text(batch_texts)
            output_node_emb[batch_nodes] = user_fea.cpu().numpy()


    print("Get item embedding ...")
    output_item_emb = np.zeros((len(item_id_set), args.embed_dim))
    for i_batch in trange(0, len(item_id_set), args.batch_size):
        batch_nodes = item_id_arr[i_batch:i_batch+args.batch_size]
        
        batch_texts = [new_dict[i] for i in batch_nodes]
        batch_texts = tokenize(batch_texts, context_length=args.context_length).to(device)

        with torch.no_grad():
            # item_fea = model.encode_image(batch_nodes, node_f, edge_index)
            item_fea = model.encode_text(batch_texts)

            output_node_emb[batch_nodes] = item_fea.cpu().numpy()



    out = []
    for i in trange(len(output_node_emb)):
        emb = list(map(lambda x: str(x), output_node_emb[i].tolist()))
        out = str(i) + "\t" + " ".join(emb)
        out.append(out)

    with open("tmp/tmp_g2p2.emb", 'w') as f:
        f.write('\n'.join(out))


    exit(1)

    


    user_dict = defaultdict(list)
    item_pool = []
    with open("tmp/MI.train") as f:
        for line in f:
            uid, iid, _ = line.rstrip().split("\t")
            item_pool.append(iid)

    with open("tmp/MI.test") as f:
        for line in f:
            uid, iid, _ = line.rstrip().split("\t")
            user_dict[uid].append(iid)
            item_pool.append(iid)


    item_pool = list(set(item_pool))
    print("Total len of item_pool", len(item_pool))
    print("Total len of user_dict", len(user_dict))

    spt_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)

    random.seed(0)
    for user in user_dict:
        if len(user_dict[user]) <= args.k_spt + args.k_val :
            continue

        random.shuffle(user_dict[user])
        spt_data[user] = list(map(int, user_dict[user][:args.k_spt]))
        val_data[user] = list(map(int, user_dict[user][args.k_spt:args.k_spt + args.k_val]))
        test_data[user] = list(map(int, user_dict[user][args.k_spt + args.k_val:args.k_spt + args.k_val + args.k_qry]))
        
    all_user_ids = list(spt_data.keys())
    random.shuffle(all_user_ids)

    print("Total len of spt_data", len(spt_data))
    print("Total len of val_data", len(val_data))
    print("Total len of test_data", len(test_data))

    

    """
        Prepare all item pools' text
    """
    candidate_texts = []
    for item in item_pool:
        item = int(item)
        candidate_texts.append(new_dict[item])



    ## ----
    model.eval()
    task_prompt = []
    for item_desc in candidate_texts:
        """
        prompt here
        """
        prompt = prompt_text + ' ' + item_desc 
        task_prompt.append(prompt)

    tmp_itempool_emb = tokenize(task_prompt, context_length=args.context_length).to(device)

    itempool_emb = []
    with torch.no_grad():
        for batch_idx in trange(0, tmp_itempool_emb.shape[0], args.batch_size):
            tmp = model.encode_text(tmp_itempool_emb[batch_idx:batch_idx + args.batch_size])
            itempool_emb.append(tmp)

    itempool_emb = torch.cat(itempool_emb, dim=0)

    output = []
    hit_count = 0
    all_test_user = list(test_data.keys())
    args.batch_size = 1
    for batch_idx in trange(0, len(all_test_user), args.batch_size):
        test_users = all_test_user[batch_idx:batch_idx + args.batch_size]
        test_users = torch.tensor([int(user) for user in test_users])

        # test_user_text_arr = new_dict[int(t_user)]
        with torch.no_grad():
            node_feas = model.encode_image(test_users, node_f, edge_index)

        """
        Data = DataHelper(arr_edge_index, args, test_users)

        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        node_feas = []
        for i_batch, sample_batched in enumerate(loader):
            s_n = sample_batched['s_n'].numpy()
            t_n = sample_batched['t_n'].numpy()
            # idx_train = sample_batched['node_idx'].to(device)
            with torch.no_grad():
                node_fea = model.encode_image(s_n, node_f, edge_index)
                node_feas.append(node_fea)
        node_feas = torch.cat(node_feas, dim=0)
        """

        similarity = torch.matmul(node_feas , itempool_emb.t())

        topk_values, topk_indices = torch.topk(similarity, 100, dim=1)

        # for i in range(topk_indices.size(0)):
            # for ans in test_data[all_test_user[i]]:
                # if ans in topk_indices[i].cpu().numpy():
                    # hit_count += 1

        ui_results = [str(all_test_user[batch_idx]), str(len(test_data[all_test_user[batch_idx]]))]
        for idx, rid in enumerate(topk_indices[0],1):
            if idx > len(test_data[all_test_user[batch_idx]]):
                break
            if rid in test_data[all_test_user[idx]]:
                ui_results.append('1')
            else:
                ui_results.append('0')
        output.append(' '.join(ui_results))


    with open("tmp/zs_g2p2.emb.ui.rec", 'w') as f:
        f.writelines("\n".join(output))

    # hit_rate = hit_count / topk_indices.size(0)

    # ans = round(np.mean(acc_list).item(), 4)
    # print('zero shot acc', ans)
    # all_acc_list[Bob].append(ans)

    # ans = round(np.mean(f1_list).item(), 4)
    # print('macro f1', ans)
    # all_macf1_list[Bob].append(ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--hidden', type=str, default=16, help='number of hidden neurons')
    parser.add_argument('--epoch_num', type=int, default=101, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=125)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--gnn_input', type=int, default=128)
    # parser.add_argument('--gnn_input', type=int, default=300)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--edge_coef', type=float, default=0.1)

    parser.add_argument('--k_spt', type=int, default=2)
    parser.add_argument('--k_val', type=int, default=1)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    parser.add_argument('--context_length', type=int, default=128) #120

    parser.add_argument('--data_name', type=str, default="Musical_Instruments")
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)  # decided by the given vocab
    # 2 heads, 6 layers, the first attempt
    # 8 heads, 12 layers, the second attempt

    args = parser.parse_args()

    start = time.perf_counter()
    all_acc_list = []
    all_macf1_list = []

        
    # the_list = ['an arts crafts or sewing of', 'arts crafts or sewing of', 'arts crafts of', 'sewing of', 'art of']
    # the_list = ['followed by a vivid description of product is ']
    the_list = ['']

    for Bob in range(len(the_list)):
        all_acc_list.append([])
        all_macf1_list.append([])
        item_prompt = the_list[Bob]
        # user_prompt = "This user says:"
        user_prompt = ""
        # print('Prompt:', prompt_text)
        for jack in range(5):
            seed = int(math.pow(2, jack))
            print('seed', seed)
            main(args)
            print('\n')

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
    exit()

    with open('./output/{}/zs_prompt_new.csv'.format(data_name), 'w') as f:
    # with open('./output/{}/zs_prompt_ttzt.csv'.format(data_name), 'w') as f:
        for i in range(len(the_list)):
            f.write('the_template=  {}'.format(the_list[i]))
            f.write('\n')
            f.write('acc')
            f.write('\n')
            for a in all_acc_list[i]:
                f.write(str(a))
                f.write(',')
            f.write('\n')
            f.write('macro-f1')
            f.write('\n')
            for a in all_macf1_list[i]:
                f.write(str(a))
                f.write(',')
            f.write('\n\n')
