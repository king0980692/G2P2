import os
import logging
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from tqdm import tqdm
import json
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string

# Enable logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)



def parse_gzip_file(path):
    with gzip.open(path, 'rb') as f:
        for line in f:
            yield json.loads(line)

def get_dataframe(path):
    data = list(parse_gzip_file(path))
    return pd.DataFrame(data)

def clean_description(df, column_names):

    for col in column_names:
        df[col] = df[col].apply(
                lambda y: np.nan if len(y) == 0 else y)

    df = df[column_names]

    df = df.replace('[]', np.nan)
    df = df.replace(' ', np.nan)

    row_idx, col_idx = np.where(pd.isnull(df))

    na_rows = np.unique(row_idx).tolist()

    meta_desc_list = df['description'].tolist()

    all_bad_set = [['']*i for i in range(1,50)] + [['N/A'], ['.'], [' ']]

    null_rows = []
    for i, desc in enumerate(df['description']):
        if desc in all_bad_set:
            null_rows.append(i)

    all_bad_idx = na_rows + null_rows

    df = df.drop(all_bad_idx)
    df['description'] = df['description'].apply(lambda x: " ".join(x))
    df.drop_duplicates(inplace=True)

    return df


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Define a function to clean and preprocess the review data
def clean_reviews(df, column_names, product_set):

    df = df[column_names]
    df = df.replace('\n', np.nan)
    df = df.replace('  ', np.nan)

    # ['reviewerID', 'asin', 'reviewText', 'summary']
    row_idx, col_idx = np.where(pd.isnull(df))
    na_rows = np.unique(row_idx).tolist()

    null_rows = []
    all_bad_set = set(['', '.', ',', 'N/A', ' ', '  ', '\n', '\t'])

    for i, review in enumerate(df['reviewText']):
        if review in all_bad_set:
            null_rows.append(i)

    all_bad_idx = na_rows + null_rows 
    return df.drop(all_bad_idx)


def filter_valid_reviews(meta_df, review_df):
    meta_prod_set = set(meta_df['asin'])

    bad_idx = []
    good_idx = []
    for i, prod in enumerate(review_df['asin']):
        if prod not in meta_prod_set:
            bad_idx.append(i)
        else:
            good_idx.append(i)

    return review_df.iloc[good_idx, :]

def prepare_w2v(meta_df, review_df):
    print("Preparing Word2Vec model...")

    """
    The Word2Vec model expects a list of sentences
    example:
        sentences = [
            ['this', 'is', 'the', 'first', 'sentence'],
            ['this', 'is', 'the', 'second', 'sentence'],
        ]
    """
    review_text = review_df['reviewText'].tolist()
    descrip_text = meta_df['description'].tolist()

    CUSTOM_FILTERS = [lambda x: x.lower()]

    # desc_corpus = [[preprocess_string(d, filters=CUSTOM_FILTERS) 
                    # for d in descrip_text[i]] 
                   # for i in range(len(descrip_text))]

    desc_corpus = [preprocess_string(d, filters=CUSTOM_FILTERS) 
                    for d in descrip_text]

    review_corpus = [preprocess_string(d, filters=CUSTOM_FILTERS) 
                     for d in review_text]

    sentences = desc_corpus + review_corpus

    # sentences = []
    # for i in range(len(desc_corpus)):
        # sentences += desc_corpus[i]
    # sentences = sentences + review_corpus

    print("Train Word2Vec model...")
    model = Word2Vec(sentences,
                     vector_size=128,
                     window=5,
                     min_count=1,
                     workers=4)

    emb_feat_list = []
    for i in range(len(desc_corpus)):
        doc = []
        for j in range(len(desc_corpus[i])):
            doc += desc_corpus[i][j]
        vec = model.wv[doc]
        vec = np.mean(vec, axis=0)
        emb_feat_list.append(vec)

    meta_emb_feat = np.array(emb_feat_list)  # .reshape(-1, 128)

    emb_feat_list = []
    for i in range(len(review_corpus)):
        vec = model.wv[review_corpus[i]]
        vec = np.mean(vec, axis=0)
        emb_feat_list.append(vec)

    review_emb_feat = np.array(emb_feat_list)  # .reshape(-1, 128)

    user_item_mapping = defaultdict(list)

    for index, reviewer_id in enumerate(review_df['reviewerID']):
        user_item_mapping[reviewer_id].append(index)

    # print('num of reviewers', len(user_item_mapping))
    id_f_map = {i: np.mean(review_emb_feat[l], axis=0).tolist()
                for i, l in user_item_mapping.items()}


    for asin, emb_feat in zip(meta_df['asin'], meta_emb_feat):
        id_f_map[asin] = emb_feat.tolist()

    # print('num of all item', len(id_f_map)-len(user_item_mapping))

    return id_f_map

def split_user_bytime(qualified_review_df, train_size=0.7):

    # 根据用户的最后一次评论时间排序
    user_last_review_time = qualified_review_df.groupby('reviewerID')['realWorldTime'].max().sort_values()

    # 按时间排序用户，计算出训练集应该有的用户数量
    num_users = len(user_last_review_time)
    num_train = int(num_users * train_size)
    train_users = user_last_review_time.iloc[:num_train].index
    test_users = user_last_review_time.iloc[num_train:].index

    # 根据用户ID分割数据集
    train_df = qualified_review_df[qualified_review_df['reviewerID'].isin(train_users)]
    test_df = qualified_review_df[qualified_review_df['reviewerID'].isin(test_users)]

    # 确保测试集的评论时间晚于训练集中的最晚时间
    early_train_time = train_df['realWorldTime'].min()
    latest_train_time = train_df['realWorldTime'].max()

    test_df = test_df[test_df['realWorldTime'] > latest_train_time]

    # 打印数据集形状和时间信息以进行验证
    print(f'Train shape: {train_df.shape}')
    print(f'Test shape: {test_df.shape}')
    print(f'Latest train time: {latest_train_time}')
    print(f'Earliest test time: {test_df["realWorldTime"].min()}')

    # 检查测试集中的用户是否出现在训练集中
    train_users_set = set(train_df['reviewerID'])
    test_users_set = set(test_df['reviewerID'])

    # 计算交集
    overlap_users = train_users_set & test_users_set

    # 输出结果
    print(f'Overlap users between train and test sets: {len(overlap_users)}')

    return train_df, test_df

def sort_by_time(df):
    copy_df = df.copy() # some warning
    copy_df['realWorldTime'] = pd.to_datetime(df['unixReviewTime'], unit='s')

    copy_df.drop('unixReviewTime', axis=1, inplace=True)

    return copy_df.sort_values(by='realWorldTime')


def get_data(file, meta_file, dataset):

    print('Loading meta data...')
    # Load and preprocess product metadata
    meta_df = get_dataframe(meta_file)
    meta_df = clean_description(meta_df, 
                                ['asin', 'title', 'description'])
    
    print('Loading data...')
    # Load and preprocess reviews
    review_df = get_dataframe(file)
    review_df = clean_reviews(
            review_df,
            ['reviewerID', 'asin', 'reviewText', 'summary', 'unixReviewTime'],
            set(meta_df['asin']))
    
    """
        Intersection of meta_df and review_df
    """
    qualified_review_df = filter_valid_reviews(meta_df, review_df)

    qualified_review_df = sort_by_time(qualified_review_df)

    train_df, test_df = split_user_bytime(qualified_review_df)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    unique_asins = train_df['asin'].unique()
    filtered_meta_df = meta_df[meta_df['asin'].isin(unique_asins)]
    assert len(filtered_meta_df) == len(unique_asins) # check the number of unique asins

    if not os.path.exists("tmp/idf_map.pkl"):

        id_f_map = prepare_w2v(filtered_meta_df, train_df)

        with open("tmp/idf_map.pkl", "wb") as f:
            pickle.dump(id_f_map, f)
    else:
        id_f_map = pickle.load(open("tmp/idf_map.pkl", "rb"))

    ## -------   Remap Feature Matrix -------

    #############
    # Train ids #
    #############
    train_ids = train_df['reviewerID'].tolist()+\
                train_df['asin'].tolist()

    train_ids = sorted(set(train_ids))

    train_id_map = {id_key: index for index, id_key in enumerate(train_ids)}
    rv_train_id_map = {index: id_key for index, id_key in enumerate(train_ids)}

    #############
    # Valid ids #
    #############
    start_index = len(train_id_map)

    val_ids = val_df['reviewerID'].tolist() + \
               val_df['asin'].tolist()  

    val_ids = sorted(set(val_ids))
    val_id_map = {id_key: index + start_index for index, id_key in enumerate(val_ids) if id_key not in train_id_map}
    rv_val_id_map = {index + start_index: id_key for index, id_key in enumerate(val_ids) if id_key not in train_id_map}


    #############
    # Test ids #
    #############
    start_index = len(train_id_map) + len(val_id_map)

    test_ids = test_df['reviewerID'].tolist() + \
               test_df['asin'].tolist()  

    test_ids = sorted(set(test_ids))
    test_id_map = {id_key: index + start_index for index, id_key in enumerate(test_ids) if id_key not in train_id_map and id_key not in val_id_map}
    rv_test_id_map = {index + start_index: id_key for index, id_key in enumerate(test_ids) if id_key not in train_id_map and id_key not in val_id_map} 

    all_id_map = {**train_id_map, **val_id_map,**test_id_map}
    rv_all_id_map = {**rv_train_id_map, **rv_val_id_map, **rv_test_id_map}


    for id, df in enumerate([train_df, val_df, test_df]):

        # Map the reviewerID and product_id to their unique integer
        s_node = list(map(all_id_map.get, df['reviewerID']))
        t_node = list(map(all_id_map.get, df['asin']))

        # Create the edges array without using range
        edge_index = np.array([s_node, t_node], dtype=int)

        # Find unique nodes involved in edges
        uniq_nodes = list(np.unique(edge_index))
        middle_idx = edge_index.shape[1]//2

        if id == 0:
            # Feature matrix preparation from w2v output
            id_fea_dict = {all_id_map[key]: value \
                           for key, value in id_f_map.items()}

            # Sort id_fea_dict by keys
            sorted_id_f_dict = OrderedDict(sorted(id_fea_dict.items()))

            fea_m = []
            for i in uniq_nodes:
                fea = sorted_id_f_dict[i]
                fea_m.append(fea)

            fea_m = np.array(fea_m, dtype=np.float32)

            # Save the feature matrix
            np.save('./tmp/{}_f_m.npy'.format(dataset), fea_m)

        ## ---------
            new_edge = [s_node + t_node, t_node + s_node]
            new_edge = np.array(new_edge)

            out_edges = []
            for idx, edge in tqdm(enumerate(edge_index.T), total=middle_idx):
                tmp_edge = str(edge[0]) + '\t' + str(edge[1]) + '\t1'
                tmp_edge = list(map(lambda x: str(x), edge.tolist()))
                tmp_edge = "\t".join(tmp_edge+['1'])
                out_edges.append(tmp_edge)

            np.save('./tmp/{}_{}_edge.npy'.format(dataset,"train"), new_edge)

            with open("tmp/MI.train.u", 'w') as f:
                f.write('\n'.join(out_edges))

        elif id == 1:

            new_edge = [s_node + t_node, t_node + s_node]
            new_edge = np.array(new_edge)

            out_edges = []
            for idx, edge in tqdm(enumerate(edge_index.T), total=middle_idx):
                tmp_edge = str(edge[0]) + '\t' + str(edge[1]) + '\t1'
                tmp_edge = list(map(lambda x: str(x), edge.tolist()))
                tmp_edge = "\t".join(tmp_edge+['1'])
                out_edges.append(tmp_edge)

            np.save('./tmp/{}_{}_edge.npy'.format(dataset,"val"), new_edge)

            with open("tmp/MI.val.u", 'w') as f:
                f.write('\n'.join(out_edges))

        else:
            user_dict = defaultdict(list)
            for idx, edge in tqdm(enumerate(edge_index.T[:middle_idx,:]), total=middle_idx):
                tmp_edge = list(map(lambda x: str(x), edge.tolist()))
                u_id = edge[0]
                i_id = edge[1]
                user_dict[u_id].append(i_id)

            # print the average user interaction
            # print(np.mean(list(map(len, user_dict.values()))))
            
            # Pick the interaction exceed 5-core for each user
            qualified_user_dict = {}
            for u_id in user_dict:
                if len(user_dict[u_id]) >= 15:
                    qualified_user_dict[u_id] = user_dict[u_id]

            # Random select 10 item from qualified_user as test set
            test_out_edges = []
            supp_out_edges = []

            test_s_nodes = []
            test_t_nodes = []
            test_s_nodes = []
            test_t_nodes = []
            for u_id, items in qualified_user_dict.items():
                supp_indices = np.random.choice(range(len(items)), 10, replace=False)
                supp_items = [items[i] for i in supp_indices]
                remaining_indices = set(range(len(items))) - set(supp_indices)
                remaining_items = [items[i] for i in remaining_indices]

                test_s_nodes += [u_id]*len(remaining_items)
                test_t_nodes += remaining_items
                test_s_nodes += [u_id]*len(supp_items)
                test_t_nodes += supp_items

                supp_out_edges += [str(u_id)+'\t'+str(i_id)+'\t1' for i_id in supp_items]
                test_out_edges += [str(u_id)+'\t'+str(i_id)+'\t1' for i_id in remaining_items]

            support_edge = [test_s_nodes + test_t_nodes, 
                            test_t_nodes + test_s_nodes]

            test_edge = [test_s_nodes + test_t_nodes, 
                         test_t_nodes + test_s_nodes]

            np.save('./tmp/{}_support_edge.npy'.format(dataset), support_edge)
            np.save('./tmp/{}_test_edge.npy'.format(dataset), test_edge)

            with open("tmp/MI.support.u", 'w') as f:
                f.write('\n'.join(test_out_edges))

            with open("tmp/MI.test.u", 'w') as f:
                f.write('\n'.join(supp_out_edges))

        ## --------- prepare id->text mapping ---------

        unique_asins = df['asin'].unique()
        # only use the asin in df
        filtered_meta_df = meta_df[meta_df['asin'].isin(unique_asins)]

        if id == 0:
            """
                user -> [review1, review2, ...] (train_df)
            """
            user_review_dict = defaultdict(list)
            for reviewer_id, review_text in zip(df['reviewerID'], df['reviewText']):
                user_review_dict[reviewer_id].append(review_text)

            id_text = {all_id_map[key]: ' '.join(texts) for key, texts in user_review_dict.items()}
            item_desc_mapping = dict(zip(filtered_meta_df['asin'], filtered_meta_df['description']))

            id_text.update({ all_id_map[key]: ' '.join(text) for key, text in item_desc_mapping.items()})

            json.dump(id_text, open('./tmp/{}_{}_text.json'.format(dataset, "train"), 'w'))

        elif id == 1:
            """
                user -> [review1, review2, ...] (train_df)
            """
            user_review_dict = defaultdict(list)
            for reviewer_id, review_text in zip(df['reviewerID'], df['reviewText']):
                user_review_dict[reviewer_id].append(review_text)

            id_text = {all_id_map[key]: ' '.join(texts) for key, texts in user_review_dict.items()}
            item_desc_mapping = dict(zip(filtered_meta_df['asin'], filtered_meta_df['description']))

            id_text.update({ all_id_map[key]: ' '.join(text) for key, text in item_desc_mapping.items()})

            json.dump(id_text, open('./tmp/{}_{}_text.json'.format(dataset, "val"), 'w'))

        else :
            """
                item -> description
            """
            supp_user_review_dict = defaultdict(list)
            test_user_review_dict = defaultdict(list)
            for reviewer_id, item_id ,review_text in zip(df['reviewerID'], df['asin'], test_df['reviewText']):

                if all_id_map[item_id] in test_t_nodes:
                    supp_user_review_dict[reviewer_id].append(review_text)
                
                if all_id_map[item_id] in test_t_nodes:
                    test_user_review_dict[reviewer_id].append(review_text)

            supp_id_text = {all_id_map[key]: ' '.join(texts) for key, texts in supp_user_review_dict.items()}

            test_id_text = {all_id_map[key]: ' '.join(texts) for key, texts in test_user_review_dict.items()}

            item_desc_mapping = dict(zip(filtered_meta_df['asin'], filtered_meta_df['description']))

            supp_id_text.update({ all_id_map[key]: ' '.join(text) for key, text in item_desc_mapping.items()})

            test_id_text.update({ all_id_map[key]: ' '.join(text) for key, text in item_desc_mapping.items()})


            json.dump(supp_id_text, open('./tmp/{}_support_text.json'.format(dataset), 'w'))
            json.dump(test_id_text, open('./tmp/{}_test_text.json'.format(dataset), 'w'))


## ----------------------------

# Main function calls
name = 'Musical_Instruments'
file = f'./data/{name}.json.gz'
meta_file = f'./data/meta_{name}.json.gz'

get_data(file, meta_file, name)

