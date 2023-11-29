import numpy as np
import json

import pandas as pd
import gzip

import gensim
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from gensim.models import word2vec
import collections


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def get_data(file, meta_file, dataset):
    df = getDF(meta_file)
    dp_meta = [0] * 4
    for i, name in enumerate(df.columns):
        if name == 'asin':
            dp_meta[0] = i
        elif name == 'title':
            dp_meta[1] = i
        elif name == 'description':
            dp_meta[2] = i
        elif name == 'category':
            dp_meta[3] = i

    wanted_df = df.iloc[:, dp_meta]
    print('wanted_df.columns', wanted_df.columns)
    # ['asin', 'title', 'description', 'category']
    for i in range(len(dp_meta)):
        b = wanted_df.iloc[:, i].apply(lambda y: np.nan if len(y) == 0 else y)
        wanted_df.iloc[:, i] = b

    wanted_df = wanted_df.replace('[]', np.nan)
    wanted_df = wanted_df.replace(' ', np.nan)
    a = wanted_df['description']
    print('wanted_df[description]', a)
    row_idx, col_idx = np.where(pd.isnull(wanted_df))

    bad_row = np.unique(row_idx).tolist()
    bad_idx = []
    b = wanted_df['description'].tolist()
    bad_set_1 = []
    for i in range(1, 50):
        a = ['']*i
        bad_set_1.append(a)

    # bad_set_1 = set(bad_set_1)
    bad_set_2 = [['N/A'], ['.'], [' ']]
    # bad_set_2 = set(bad_set_2)
    all_bad_set = bad_set_1 + bad_set_2
    print(all_bad_set)

    for i in range(len(b)):
        # print('wanted_desc[{}]'.format(i), wanted_desc[i])
        # if b[i] == [''] or b[i] == ['', ''] or b[i] == ['', '', ''] or b[i] == ['', '', '', ''] or \
        #         b[i] == ['N/A'] or b[i] == ['.'] or b[i] == [' '] or b[i]==['', '', '', '', '', '', '', '']:
        if b[i] in all_bad_set:
            bad_idx.append(i)

    print('bad_idx', bad_idx)
    all_bad_idx = bad_row + bad_idx
    meta_full_df = wanted_df.drop(all_bad_idx)
    print('shape after the drop', meta_full_df.shape[0])

    df = getDF(file)
    dp_review = [0] * 4
    for i, name in enumerate(df.columns):
        if name == 'reviewerID':
            dp_review[0] = i
        elif name == 'asin':
            dp_review[1] = i
        elif name == 'reviewText':
            dp_review[2] = i
        elif name == 'summary':
            dp_review[3] = i

    wanted_df = df.iloc[:, dp_review]
    wanted_df = wanted_df.replace('\n', np.nan)
    wanted_df = wanted_df.replace('  ', np.nan)
    # wanted_df = wanted_df.replace('\t', np.nan)
    print('wanted_df.columns', wanted_df.columns)
    # ['reviewerID', 'asin', 'reviewText', 'summary']
    row_idx, col_idx = np.where(pd.isnull(wanted_df))
    bad_row = np.unique(row_idx).tolist()
    bad_idx = []
    bad_set = set(['', '.', ',', 'N/A', ' ', '  ', '\n', '\t'])
    b = wanted_df['reviewText'].tolist()
    for i in range(len(b)):
        if b[i] in bad_set:
            bad_idx.append(i)
    all_bad_idx = bad_row + bad_idx
    review_full_df = wanted_df.drop(all_bad_idx)
    print('review_full_df.shape', review_full_df.shape)

    products = meta_full_df['asin'].tolist()
    prod_set = set(products)

    asin = review_full_df.iloc[:, 1].to_numpy()
    print(asin)

    bad_idx = []
    good_idx = []
    for i in range(review_full_df.shape[0]):
        if asin[i] not in prod_set:
            bad_idx.append(i)
        else:
            good_idx.append(i)

    qualify_review = review_full_df.iloc[good_idx, :]


    review_text = qualify_review['reviewText'].tolist()

    descrip = meta_full_df['description'].tolist()

    CUSTOM_FILTERS = [lambda x: x.lower()]
    process_1 = [[preprocess_string(d, filters=CUSTOM_FILTERS) for d in descrip[i]] for i in range(len(descrip))]
    # process_1 = descrip
    process_2 = [preprocess_string(d, filters=CUSTOM_FILTERS) for d in review_text]

    sentences = []
    for i in range(len(process_1)):
        sentences += process_1[i]

    sentences = sentences + process_2
    print('len(sentences)', len(sentences))

    model = word2vec.Word2Vec(sentences, workers=3, vector_size=128, min_count=1, window=5, sg=0)

    emb_feat_list = []
    for i in range(len(process_1)):
        doc = []
        for j in range(len(process_1[i])):
            doc += process_1[i][j]
        vec = model.wv[doc]
        vec = np.mean(vec, axis=0)
        emb_feat_list.append(vec)

    meta_emb_feat = np.array(emb_feat_list)  # .reshape(-1, 128)
    print('emb_feat of product', meta_emb_feat[:1])

    emb_feat_list = []
    for i in range(len(process_2)):
        vec = model.wv[process_2[i]]
        vec = np.mean(vec, axis=0)
        emb_feat_list.append(vec)

    review_emb_feat = np.array(emb_feat_list)  # .reshape(-1, 128)
    print('emb_feat of review', review_emb_feat[:1])

    ## ===========

    reviewerID = qualify_review['reviewerID'].tolist()
    id_dic_list = collections.defaultdict(list)

    for i in range(len(reviewerID)):
        id_dic_list[reviewerID[i]].append(i)

    id_f_map = {}
    for i in id_dic_list:
        l = id_dic_list[i]
        fea = review_emb_feat[l]
        id_f_map[i] = np.mean(fea, axis=0).tolist()
    print('number of reviewers', len(id_f_map))
    asin = meta_full_df['asin'].tolist()
    for j in range(len(asin)):
        id_f_map[asin[j]] = meta_emb_feat[j].tolist()
    print('num of all node level text', len(id_f_map))

    # json.dump(id_f_map, open('./data/amazon/{}_id_f_map.json'.format(dataset), 'w'))
    descrip = meta_full_df['description'].tolist()
    asin = meta_full_df['asin'].tolist()
    review_text = qualify_review['reviewText'].tolist()
    reviewerID = qualify_review['reviewerID'].tolist()
    asin_dict = {}
    for i in range(len(asin)):
        asin_dict[asin[i]] = descrip[i]

    product_id = qualify_review['asin'].tolist()
    id_list = reviewerID + product_id + asin

    id_set = set(id_list)
    id_set = list(id_set)

    id_map = {id_set[i]: i for i in range(len(id_set))}

    s_node = [id_map[i] for i in reviewerID]
    t_node = [id_map[i] for i in product_id]

    edge = [s_node, t_node]
    edge = np.array(edge, dtype=int)

    edge_node = np.unique(edge)
    edge_node = edge_node.tolist()
    print('number of node texts', len(edge_node))

    er_view_dict = collections.defaultdict(list)
    for i in range(len(reviewerID)):
        er_view_dict[reviewerID[i]].append(review_text[i])

    id_text = {}
    for i in er_view_dict:
        id_text[id_map[i]] = ' '.join(er_view_dict[i])

    for j in asin_dict:
        id_text[id_map[j]] = ' '.join(asin_dict[j])

    id_fea_dict = {}
    for i in id_f_map:
        id_fea_dict[id_map[i]] = id_f_map[i]

    fea_m = []
    sorted_id_f_dict = collections.OrderedDict(sorted(id_fea_dict.items()))
    for i in edge_node:
        fea = sorted_id_f_dict[i]
        fea_m.append(fea)

    fea_m = np.array(fea_m, dtype=np.float32)

    np.save('./tmp/{}_f_m.npy'.format(dataset), fea_m)

    edge_node_map = {}
    for i in range(len(edge_node)):
        edge_node_map[edge_node[i]] = i

    new_s_node = [edge_node_map[i] for i in s_node]
    new_t_node = [edge_node_map[i] for i in t_node]

    new_edge = [new_s_node + new_t_node, new_t_node + new_s_node]
    new_edge = np.array(new_edge)
    print('num of nodes', len(edge_node))
    print('num of edges', new_edge.shape[1])
    print('num of labels', meta_full_df.shape[0])

    np.save('./tmp/{}_edge.npy'.format(dataset), new_edge)

    node_texts = []
    lenth_list = []
    for i in edge_node:
        node_texts.append(id_text[i])
        a = id_text[i].split()
        lenth = len(a)
        lenth_list.append(lenth)

    print('number of node texts', len(node_texts))  # 500386
    print('average context length', round(np.mean(lenth_list), 2))

    # np.save('./data/amazon/appliances_text.npy', node_texts)
    text_dict = {}
    for i in range(len(node_texts)):
        text_dict[i] = node_texts[i]

    json.dump(text_dict, open('./tmp/{}_text.json'.format(dataset), 'w'))

    lable_id = meta_full_df['asin'].tolist()

    first_id = []
    for i in lable_id:
        first_id.append(id_map[i])
    print('lenth of first_id', len(first_id))
    second_id = []
    good = []

    for i in range(len(first_id)):
        if first_id[i] in edge_node_map:
            second = edge_node_map[first_id[i]]
            second_id.append(second)
            good.append(i)
    print('length of second_id, number of labels', len(second_id))

    wanted_lables = meta_full_df['category'].iloc[good].tolist()

    final_lables = []
    for i in wanted_lables:
        lable = i[1:]
        lable = ' '.join(lable)
        # lable = preprocess_string(lable)
        # lable = ' '.join(lable)
        final_lables.append(lable)
    print('final_lables[:10]', final_lables[:10])

    label_arr = np.array(final_lables)
    label_uniq, label_count = np.unique(label_arr, return_counts=True)
    print('number of label category:', label_uniq.shape[0], '\t label_count', label_count)

    labels_dic = {}
    for i in range(len(second_id)):
        labels_dic[second_id[i]] = final_lables[i]

    json.dump(labels_dic, open('./tmp/{}_id_labels.json'.format(dataset), 'w'))

## ===============================

name = 'Musical_Instruments' # 'Video_Games'
# name = 'Arts_Crafts_and_Sewing'
file = './data/{}.json.gz'.format(name)
meta_file = './data/meta_{}.json.gz'.format(name)

get_data(file, meta_file, name)
