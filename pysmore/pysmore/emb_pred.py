import sys
import gzip
import json
import argparse
import random
import math
import concurrent.futures
from collections import defaultdict
from math import log, sqrt
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseManager, DictProxy
from multiprocessing import Pool, Manager, freeze_support

import faiss
from annoy import AnnoyIndex


def eu_distance(v, v2):
    try:
        return -(sum((a-b)**2 for a, b in zip(v, v2)))  # omit sqrt
    except:
        return -100.


def cosine_distance(v, v2):
    try:
        score = sum((a*b) for a, b in zip(v, v2)) / \
            (sqrt(sum(a*a for a in v)*sum(b*b for b in v2)))
        if math.isnan(score):
            return 0
        return score
    except:
        return 0


def dot_sim(v, v2):
    try:
        score = sum((a*b) for a, b in zip(v, v2))
        if math.isnan(score):
            return 0
        return score
    except:
        return 0


# ------

def process_ui_query(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    global embed_dim
    item_pool = list(iids.keys()) + list(cold_iids.keys())

    ui_results = [uid, str(len(test_ui[uid]))]

    # scoring
    ui_scores = defaultdict(lambda: 0.)
    for rid in item_pool:
        if rid in train_ui[uid]:
            continue
        if uid in embed and rid in embed:
            ui_scores[rid] += sim(embed[uid], embed[rid])  # dot product
        else:
            ui_scores[rid] += 0.

    # ranking
    for rid in sorted(ui_scores, key=ui_scores.get, reverse=True)[:len(test_ui[uid])]:
        if rid in test_ui[uid]:
            ui_results.append('1')  # hit or not
        else:
            ui_results.append('0')

    return ' '.join(ui_results)


def process_ui_query2(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    global embed_dim

    item_pool = list(iids.keys()) + list(cold_iids.keys())
    observed_items = list(train_ui[uid].keys())

    ui_results = [uid, str(len(test_ui[uid]))]  # uid, total_len

    # scoring_matrix
    q_vec = np.array(embed[uid]) if uid in embed else np.zeros((embed_dim,))

    pool_vec = np.array([np.array(embed[it])
                         if it in embed and it not in observed_items
                         else np.zeros((embed_dim,)) for it in item_pool])

    scores = pool_vec @ q_vec

    top_k_list = np.array(item_pool)[np.argsort(scores)[::-1]].tolist()

    # Evaluation
    for rid in top_k_list[:len(test_ui[uid])]:
        # for rid in sorted(scores, key=ui_scores.get, reverse=True):
        if rid in test_ui[uid]:
            ui_results.append('1')  # hit or not
        else:
            ui_results.append('0')

    return ' '.join(ui_results)


def process_ui_query3(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    global embed_dim

    item_pool = list(iids.keys()) + list(cold_iids.keys())

    # scoring_matrix
    q_array = []
    ui_results_dict = {}
    pool_array = []
    for u in uid:
        q_vec = np.array(embed[u]) if u in embed else np.zeros((embed_dim,))
        observed_items = list(train_ui[u].keys())
        ui_results_dict[u] = [u, str(len(test_ui[u]))]  # uid, total_len
        pool_vec = np.array([np.array(embed[it]) if it in embed and it not in observed_items else np.zeros(
            (embed_dim,)) for it in item_pool])

        q_array.append(q_vec)
        pool_array.append(pool_vec)

    q_array = np.vstack(q_array)
    pool_array = np.vstack(pool_vec)

    scores = pool_array @ q_array.T

    top_k_array = []
    # top_k_list = np.array(item_pool)[np.argsort(scores)[::-1]].tolist()[:len(test_ui[uid])]
    for id, u in enumerate(uid):
        top_k_list = np.array(item_pool)[np.argsort(scores[:, id])[
            ::-1]].tolist()[:len(test_ui[u])]
        top_k_array.append(top_k_list)

    # Evaluation
    for id, u in enumerate(uid):
        for rid in top_k_array[id]:
            # for rid in sorted(scores, key=ui_scores.get, reverse=True):
            if rid in test_ui[u]:
                ui_results_dict[u].append('1')  # hit or not
            else:
                ui_results_dict[u].append('0')

    return '\n'.join([' '.join(u) for u in ui_results_dict.values()])


def process_ii_query3(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    global embed_dim
    item_pool = list(iids.keys()) + list(cold_iids.keys())

    q_array = []
    ui_results_dict = {}
    pool_array = []
    for u in uid:
        q_vec = np.sum([embed[i] if i in embed else np.zeros(
            (embed_dim,)) for i in train_ui[u]], axis=0)
        # q_vec = np.array(embed[u]) if u in embed else np.zeros((args.emb_dim,))
        observed_items = list(train_ui[u].keys())
        ui_results_dict[u] = [u, str(len(test_ui[u]))]  # uid, total_len
        pool_vec = np.array([np.array(embed[it]) if it in embed and it not in observed_items else np.zeros(
            (embed_dim,)) for it in item_pool])

        q_array.append(q_vec)
        pool_array.append(pool_vec)

    # scoring
    q_array = np.vstack(q_array)
    pool_array = np.vstack(pool_vec)

    scores = pool_array @ q_array.T

    top_k_array = []
    # top_k_list = np.array(item_pool)[np.argsort(scores)[::-1]].tolist()[:len(test_ui[uid])]
    for id, u in enumerate(uid):
        top_k_list = np.array(item_pool)[np.argsort(scores[:, id])[
            ::-1]].tolist()[:len(test_ui[u])]
        top_k_array.append(top_k_list)

    # Evaluation
    for id, u in enumerate(uid):
        for rid in top_k_array[id]:
            # for rid in sorted(scores, key=ui_scores.get, reverse=True):
            if rid in test_ui[u]:
                ui_results_dict[u].append('1')  # hit or not
            else:
                ui_results_dict[u].append('0')

    return '\n'.join([' '.join(u) for u in ui_results_dict.values()])


def process_ii_query2(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    global embed_dim
    item_pool = list(iids.keys()) + list(cold_iids.keys())
    observed_items = list(train_ui[uid].keys())

    ui_results = [uid, str(len(test_ui[uid]))]

    q_vec = np.sum([embed[i] for i in train_ui[uid]], axis=0)

    pool_vec = np.array([np.array(embed[it]) if it in embed and it not in observed_items else np.zeros(
        (embed_dim,)) for it in item_pool])

    # scoring
    scores = pool_vec @ q_vec

    top_k_list = np.array(item_pool)[np.argsort(
        scores)[::-1]].tolist()[:len(test_ui[uid])]

    # ranking
    for rid in top_k_list:
        if rid in test_ui[uid]:
            ui_results.append('1')
        else:
            ui_results.append('0')

    return ' '.join(ui_results)


def process_ii_query(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    item_pool = list(iids.keys()) + list(cold_iids.keys())

    q_embed = np.sum([embed[i] for i in train_ui[uid]], axis=0)
    ui_results = [uid, str(len(test_ui[uid]))]

    # scoring
    ui_scores = defaultdict(lambda: 0.)
    for rid in item_pool:
        if rid in train_ui[uid]:
            continue
        if uid in embed and rid in embed:
            ui_scores[rid] += sim(q_embed, embed[rid])
        else:
            ui_scores[rid] += 0.

    # len(ui_score) = |item_pool| - ui_interactions

    # ranking
    for rid in sorted(ui_scores, key=ui_scores.get, reverse=True)[:len(test_ui[uid])]:
        if rid in test_ui[uid]:
            ui_results.append('1')
        else:
            ui_results.append('0')

    return ' '.join(ui_results)


def initializer(opts):
    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    global embed_dim

    sim = opts[0]
    iids = opts[1]
    cold_iids = opts[2]
    train_ui = opts[3]
    test_ui = opts[4]
    embed = opts[5]
    embed_dim = opts[6]


def process_args():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--train', help='data.ui.train')
    parser.add_argument('--test', help='data.ui.test')
    parser.add_argument('--embed', help='embeddding file')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='emedding demensions')
    parser.add_argument('--cold_user', type=int, default=0,
                        help='to test cold user')
    parser.add_argument('--cold_item', type=int, default=0,
                        help='to test cold item')
    parser.add_argument('--num_test', type=int,
                        default=sys.maxsize, help='# of sampled tests')
    parser.add_argument('--worker', type=int, default=1, help='# of threads')
    parser.add_argument('--query', type=str, choices=['warm', 'cold', 'all'], default='all', help='query type')

    parser.add_argument(
        '--sim', choices=['dot', 'cosine'], default='dot', help='sim metric')
    parser.add_argument('--faiss', action='store_true',
                        help='use faiss to compute similarities')
    parser.add_argument('--annoy', action='store_true',
                        help='use annoy to compute similarities')

    args = parser.parse_args()

    return args


def recommendations():

    args = process_args()

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    global embed_dim

    embed_dim = args.embed_dim

    sim = {'dot': dot_sim, 'cosine': cosine_distance}[args.sim]
    print('load train data from', args.train)

    # Read Train Data
    uids, iids = {}, {}
    train_ui = defaultdict(dict)
    train_counter = 0.

    uid_map, iid_map = {}, {}
    rv_uid_map, rv_iid_map = {}, {}

    with open(args.train) as f:
        for line in tqdm(f.readlines()):
            uid, iid, target = line.rstrip().split('\t')[:3]
            if float(target) <= 0:
                continue
            train_counter += 1.
            uids[uid] = 1
            iids[iid] = 1
            train_ui[uid][iid] = 1

            if uid not in uid_map:
                uid_map[uid] = len(uid_map)
                rv_uid_map[len(uid_map)-1] = uid

            if iid not in iid_map:
                iid_map[iid] = len(iid_map)
                rv_iid_map[len(iid_map)-1] = iid

    # Read Test Data
    print('load test data from', args.test)
    cold_uids, cold_iids = {}, {}
    test_counter = 0.
    test_ui = defaultdict(dict)
    with open(args.test) as f:
        for line in tqdm(f.readlines()):
            uid, iid, target = line.rstrip().split('\t')[:3]
            if float(target) <= 0:
                continue
            if uid not in uids:
                cold_uids[uid] = 1
            if iid not in iids:
                cold_iids[iid] = 1

            # if iid in cold_iids and not args.cold_item:
                # continue

            ##################
            # Cold item case #
            ##################
            if iid not in iid_map:
                iid_map[iid] = len(iid_map)
                rv_iid_map[len(iid_map)-1] = iid

            test_counter += 1.
            test_ui[uid][iid] = float(target)

    # Load Embedding
    print("load embeddings from", args.embed)
    embed = {}
    embedding_matrix = np.zeros((len(iid_map), args.embed_dim))
    with open(args.embed, 'r') as f:
        lines = f.readlines()
        for line in lines[:]:
            line = line.rstrip().split('\t')
            ID = line[0]
            try:
                embed[ID] = list(map(float, line[1].split(' ')))
            except:
                breakpoint()

            if ID in iid_map:
                embedding_matrix[iid_map[ID]] = np.fromstring(line[1], sep=' ')
    print("The shape of embedding matrix is: ", embedding_matrix.shape)

    # ------------------------------------------------------------------

    print('num of warm user:', len(uids))
    print('num of cold user:', len(cold_uids))

    print('num of warm item:', len(iids))
    print('num of cold item:', len(cold_iids))

    warm_u_queries = [u for u in test_ui if u not in cold_uids]
    cold_u_queries = [u for u in test_ui if u in cold_uids]
    print('warm user query:', len(warm_u_queries))
    print('cold user query:', len(cold_u_queries))

    # print('avg. train item per user:', train_counter/len(train_ui))
    # print('avg. test item per user:', test_counter/len(test_ui))

    print("Using %s queries" % args.query)
    if args.query == 'all':
        queries = warm_u_queries + cold_u_queries
    else:
        queries = warm_u_queries if args.query == 'warm' else cold_u_queries

    print("Item Pool Size :", len(iids) + len(cold_iids))
    print("Item Pool Size :", embedding_matrix.shape[0])
    print("Item Pool Size :", len(iid_map))
    print("Total Queries :", len(queries))

    if args.annoy:

        annoy_index = AnnoyIndex(args.embed_dim)
        for i in range(embedding_matrix.shape[0]):
            v = embedding_matrix[i]
            annoy_index.add_item(i, v)

        annoy_index.build(10)  # 10 trees

        rec_ui = []
        for uid in tqdm(queries[:args.num_test]):

            u_repr = np.array(embed[uid])
            rec = annoy_index.get_nns_by_vector(u_repr, 1000)

            # item_pool = list(iids.keys()) + list(cold_iids.keys())
            observed_items = list(train_ui[uid].keys())

            ui_results = [uid, str(len(test_ui[uid]))]  # uid, total_len
            top_k_list = [rv_iid_map[r]
                          for r in rec if r not in observed_items][:len(test_ui[uid])]

            # Evaluation
            for rid in top_k_list:
                # for rid in sorted(scores, key=ui_scores.get, reverse=True):
                if rid in test_ui[uid]:
                    ui_results.append('1')  # hit or not
                else:
                    ui_results.append('0')

            rec_ui.append(' '.join(ui_results))

        print('write the result to', args.embed+'.ui.rec')
        with open(args.embed+'.ui.rec', 'w') as f:
            f.write('%s\n' % ('\n'.join(rec_ui)))

    elif args.faiss:

        embedding_matrix = embedding_matrix.astype(np.float32)
        faiss_index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        faiss_index.add(embedding_matrix)

        rec_ui = []
        predictions = defaultdict(list)

        # query_vec = np.array([np.array(embed[uid]) for uid in queries])

        # Do top-k query by faiss
        # _, rec = faiss_index.search(query_vec.astype(np.float32), 1000)

        for uid in tqdm(queries[:args.num_test]):
            _, rec = faiss_index.search(
                np.expand_dims(np.array(embed[uid]), axis=0).astype(
                    np.float32),
                1000
            )

            # item_pool = list(iids.keys()) + list(cold_iids.keys())
            observed_items = list(train_ui[uid].keys())

            ui_results = [uid, str(len(test_ui[uid]))]  # uid, total_len
            top_k_list = [rv_iid_map[r] for r in rec[0]
                          if r not in observed_items][:len(test_ui[uid])]
            predictions[uid] = top_k_list

            # Evaluation
            for rid in top_k_list:
                # for rid in sorted(scores, key=ui_scores.get, reverse=True):
                if rid in test_ui[uid]:
                    ui_results.append('1')  # hit or not
                else:
                    ui_results.append('0')

            rec_ui.append(' '.join(ui_results))

        print('write the result to', args.embed+'.ui.rec')
        with open(args.embed+'.ui.rec', 'w') as f:
            f.write('%s\n' % ('\n'.join(rec_ui)))
        # with open(args.embed+'.ui.pred', 'w') as f:
            # f.write('%s\n' % ('\n'.join(['%s %s' % (uid, ','.join([str(iids[r]) for r in rec[0]])) for uid, rec in predictions.items()])))
    else:
        # U2I
        random.shuffle(queries)
        queries = queries[:args.num_test]
        # warm_u_queries = warm_u_queries[:]
        # rec_ui = []
        # def divide_chunks(l, size):
        # for i in range(0, len(l), size):
        # yield l[i:i+size]
        # warm_u_queries_batch = list(divide_chunks(warm_u_queries, 20))

        # '547516' / '007c34d1-d573-2fb9-979f-c22062d1fe88'

        """
        rec_ui = process_ui_query2('007c34d1-d573-2fb9-979f-c22062d1fe88')
        
        for user in warm_u_queries:
            rec_ui = process_ui_query2(user)
            break
        """

        with mp.get_context('spawn').Pool(processes=args.worker, initializer=initializer, initargs=((sim, iids, cold_iids, train_ui, test_ui, embed, args.embed_dim),)) as pool:
            # pool.map(process_ui_query2, warm_u_queries )
            # r = list(tqdm(p.imap(_foo, range(30)), total=30))

            rec_ui = list(
                tqdm(pool.imap(process_ui_query, queries), total=len(queries)))
            # rec_ui = list(tqdm(pool.imap(process_ui_query2, warm_u_queries), total=len(warm_u_queries) ))
            # rec_ui = list(tqdm(pool.imap(process_ui_query3, warm_u_queries_batch), total=len(warm_u_queries_batch) ))

        # with concurrent.futures.ProcessPoolExecutor(max_workers=args.worker) as executor:
            # for res in tqdm(executor.map(process_ui_query2, warm_u_queries)):
            # rec_ui.append(res)
        print('write the result to', args.embed+'.ui.rec')
        with open(args.embed+'.ui.rec', 'w') as f:
            f.write('%s\n' % ('\n'.join(rec_ui)))

        exit(0)

        # I2I
        random.shuffle(queries)
        queries = queries[:args.num_test]
        # warm_u_queries = warm_u_queries[:]
        rec_ii = []

        warm_u_queries_batch = list(divide_chunks(queries, 20))
        with mp.get_context('spawn').Pool(processes=args.worker, initializer=initializer, initargs=((sim, iids, cold_iids, train_ui, test_ui, embed, args.embed_dim),)) as pool:
            # pool.map(process_ui_query2, warm_u_queries )
            # r = list(tqdm(p.imap(_foo, range(30)), total=30))

            rec_ii = list(
                tqdm(pool.imap(process_ii_query2, queries), total=len(queries)))
            # rec_ii = list(tqdm(pool.imap(process_ii_query3, warm_u_queries_batch), total=len(warm_u_queries_batch) ))
        # with concurrent.futures.ProcessPoolExecutor(max_workers=args.worker) as executor:
            # for res in tqdm(executor.map(process_ii_query2, warm_u_queries)):
            # rec_ii.append(res)
        print('write the result to', args.embed+'.ii.rec')
        with open(args.embed+'.ii.rec', 'w') as f:
            f.write('%s\n' % ('\n'.join(rec_ii)))


if __name__ == '__main__':
    recommendations()
