import ipdb
import argparse
import os
import logging
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from tqdm import tqdm, trange
import json
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string

# Enable logging
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument(
    "--dataset",
    type=str,
    default="Musical_Instruments",
)
args = parser.parse_args()

data_abbr = {
    "Musical_Instruments": "MI",
    "All_Beauty": "BE",
    "Industrial_and_Scientific": "IS",
    "Sports_and_Outdoors": "SO",
    "Toys_and_Games": "TG",
    "Arts_Crafts_and_Sewing": "AC",
    "reviews_Beauty_5": "BE5",
    "reviews_Musical_Instruments_5": "MI5",
}


def parse_gzip_file(path):
    load_fn = json.loads if "reviews" not in path else eval
    with gzip.open(path, "rb") as f:
        for line in f:
            # yield json.loads(line)
            yield load_fn(line)


def get_dataframe(path):
    data = list(parse_gzip_file(path))
    return pd.DataFrame(data)


def clean_description(df, column_names):
    df = df[column_names]

    for col in column_names:
        # filter nan
        df = df[df[col].notna()].reset_index(drop=True)

        df[col] = df[col].apply(lambda y: np.nan if len(y) == 0 else y)

    df = df.replace("[]", np.nan)
    df = df.replace(" ", np.nan)
    df = df.replace("", np.nan)

    row_idx, col_idx = np.where(pd.isnull(df))

    na_rows = np.unique(row_idx).tolist()

    # if type(df["description"][0]) == str:
    #     df["description"] = df["description"].apply(lambda x: " ".join(x)])
    # meta_desc_list = df["description"].tolist()

    all_bad_set = [[""] * i for i in range(1, 50)] + [["N/A"], ["."], [" "]]

    null_rows = []
    for i, desc in enumerate(df["description"]):
        if desc in all_bad_set:
            null_rows.append(i)

    all_bad_idx = na_rows + null_rows

    df = df.drop(all_bad_idx)

    if type(df["description"].iloc[0]) == list:
        df["description"] = df["description"].apply(lambda x: " ".join(x))
    df.drop_duplicates(inplace=True)

    return df


def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


# Define a function to clean and preprocess the review data


def clean_reviews(df, column_names, product_set):
    df = df[column_names]
    df = df.replace("\n", np.nan)
    df = df.replace("  ", np.nan)

    # ['reviewerID', 'asin', 'reviewText', 'summary']
    row_idx, col_idx = np.where(pd.isnull(df))
    na_rows = np.unique(row_idx).tolist()

    null_rows = []
    all_bad_set = set(["", ".", ",", "N/A", " ", "  ", "\n", "\t"])

    for i, review in enumerate(df["reviewText"]):
        if review in all_bad_set:
            null_rows.append(i)

    all_bad_idx = na_rows + null_rows
    return df.drop(all_bad_idx)


def filter_valid_reviews(meta_df, review_df):
    meta_prod_set = set(meta_df["asin"])

    bad_idx = []
    good_idx = []
    for i, prod in enumerate(review_df["asin"]):
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
    review_text = review_df["reviewText"].tolist()
    descrip_text = meta_df["description"].tolist()

    CUSTOM_FILTERS = [lambda x: x.lower()]

    # desc_corpus = [[preprocess_string(d, filters=CUSTOM_FILTERS)
    # for d in descrip_text[i]]
    # for i in range(len(descrip_text))]

    desc_corpus = [preprocess_string(d, filters=CUSTOM_FILTERS) for d in descrip_text]

    review_corpus = [preprocess_string(d, filters=CUSTOM_FILTERS) for d in review_text]

    sentences = desc_corpus + review_corpus

    print("Train Word2Vec model...")
    model = Word2Vec(
        sentences, vector_size=128, window=5, min_count=1, workers=4, epochs=5
    )

    emb_feat_list = []
    for i in trange(len(desc_corpus)):
        doc = []
        for j in range(len(desc_corpus[i])):
            doc.append(desc_corpus[i][j])
        vec = model.wv[doc]
        vec = np.mean(vec, axis=0)
        emb_feat_list.append(vec)

    meta_emb_feat = np.array(emb_feat_list)  # .reshape(-1, 128)

    emb_feat_list = []
    for i in trange(len(review_corpus)):
        if len(review_corpus[i]) == 0:
            vec = np.zeros(128)
            print(i, "empty review")
        else:
            vec = model.wv[review_corpus[i]]
            vec = np.mean(vec, axis=0)
        emb_feat_list.append(vec)

    review_emb_feat = np.array(emb_feat_list)  # .reshape(-1, 128)

    user_item_mapping = defaultdict(list)

    for index, reviewer_id in enumerate(review_df["reviewerID"]):
        user_item_mapping[reviewer_id].append(index)

    print("num of reviewers", len(user_item_mapping))
    id_f_map = {
        i: np.mean(review_emb_feat[l], axis=0).tolist()
        for i, l in user_item_mapping.items()
    }

    for asin, emb_feat in zip(meta_df["asin"], meta_emb_feat):
        id_f_map[asin] = emb_feat.tolist()

    print("num of all item", len(id_f_map) - len(user_item_mapping))

    return id_f_map


def split_user_bytime(qualified_review_df, train_size=0.7):
    # 根据用户的最后一次评论时间排序
    user_last_review_time = (
        qualified_review_df.groupby("reviewerID")["realWorldTime"].max().sort_values()
    )

    # 按时间排序用户，计算出训练集应该有的用户数量
    num_users = len(user_last_review_time)
    num_train = int(num_users * train_size)
    train_users = user_last_review_time.iloc[:num_train].index
    test_users = user_last_review_time.iloc[num_train:].index

    # 根据用户ID分割数据集
    train_df = qualified_review_df[qualified_review_df["reviewerID"].isin(train_users)]

    test_df = qualified_review_df[qualified_review_df["reviewerID"].isin(test_users)]

    # 确保测试集的评论时间晚于训练集中的最晚时间
    early_train_time = train_df["realWorldTime"].min()
    latest_train_time = train_df["realWorldTime"].max()

    test_df = test_df[test_df["realWorldTime"] > latest_train_time]

    # 打印数据集形状和时间信息以进行验证
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Latest train time: {latest_train_time}")
    print(f'Earliest test time: {test_df["realWorldTime"].min()}')

    # 检查测试集中的用户是否出现在训练集中
    train_users_set = set(train_df["reviewerID"])
    test_users_set = set(test_df["reviewerID"])

    # 计算交集
    overlap_users = train_users_set & test_users_set

    # 输出结果
    print(f"Overlap users between train and test sets: {len(overlap_users)}")

    return train_df, test_df


def sort_by_time(df):
    copy_df = df.copy()  # some warning
    copy_df["realWorldTime"] = pd.to_datetime(df["unixReviewTime"], unit="s")

    # copy_df.drop('unixReviewTime', axis=1, inplace=True)

    return copy_df.sort_values(by="realWorldTime")


def extract_from_df(qualified_review_df, cat_dict, datamaps):
    qualified_review_df["category"] = qualified_review_df["asin"].map(cat_dict)

    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}

    for iid, attributes in qualified_review_df[["asin", "category"]].values:
        if len(attributes) == 1:
            continue
        if iid not in datamaps["item2id"]:
            continue
        item_id = datamaps["item2id"][iid]
        items2attributes[item_id] = []

        for attribute in attributes[1:]:  # skip the first category
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])

    datamaps["attribute2id"] = attribute2id
    datamaps["id2attribute"] = id2attribute
    datamaps["attributeid2num"] = attributeid2num
    return datamaps


def sample_test_data(dataset, data_path, test_num=99, sample_type="random"):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """

    test_file = f"negative_samples.txt"

    item_count = defaultdict(int)
    user_items = defaultdict()

    lines = open(data_path).readlines()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        user_items[user] = items
        for item in items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    user_neg_items = defaultdict()

    print("Creating negative samples...")
    for user, user_seq in tqdm(user_items.items()):
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == "random":
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else:  # sample_type == 'pop':
                sample_ids = np.random.choice(
                    all_item, test_num, replace=False, p=probability
                )
            sample_ids = [
                str(item)
                for item in sample_ids
                if item not in user_seq and item not in test_samples
            ]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        user_neg_items[user] = test_samples

    print("./tmp/{}_".format(data_abbr[dataset]) + test_file)
    with open("./tmp/{}_".format(data_abbr[dataset]) + test_file, "w") as out:
        for user, samples in user_neg_items.items():
            out.write(user + " " + " ".join(samples) + "\n")


def create_seq_data(df, id_map, rv_id_map, dataset):
    df["uid"] = df["reviewerID"].apply(lambda x: str(id_map[x]))
    df["iid"] = df["asin"].apply(lambda x: str(id_map[x]))

    # Group by user and item
    user_items = df.groupby("uid")["iid"].apply(list).to_dict()

    with open(f"tmp/{data_abbr[dataset]}_sequential.txt", "w") as out:
        for user, items in user_items.items():
            out.write(user + " " + " ".join(items) + "\n")

    sample_test_data(dataset, f"tmp/{data_abbr[dataset]}_sequential.txt")

    return user_items


def get_data(file, meta_file, dataset):
    print(f"Loading meta data...{meta_file}")
    # Load and preprocess product metadata
    meta_df = get_dataframe(meta_file)
    if "category" in meta_df.columns:
        cat_dict = dict(zip(meta_df["asin"], meta_df["category"]))
    elif "categories" in meta_df.columns:
        cat_dict = dict(zip(meta_df["asin"], meta_df["categories"]))
    else:
        raise ValueError("No category column found in meta_df")
    meta_df = clean_description(meta_df, ["asin", "title", "description"])

    print(f"Loading data...{file}")
    # Load and preprocess reviews
    ori_review_df = get_dataframe(file)

    review_df = clean_reviews(
        ori_review_df,
        ["overall", "reviewerID", "asin", "reviewText", "summary", "unixReviewTime"],
        set(meta_df["asin"]),
    )

    """
        Intersection of meta_df and review_df
    """
    qualified_review_df = filter_valid_reviews(meta_df, review_df)

    qualified_review_df = sort_by_time(qualified_review_df)

    train_df, test_df = split_user_bytime(qualified_review_df)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    unique_asins = train_df["asin"].unique()
    filtered_meta_df = meta_df[meta_df["asin"].isin(unique_asins)]
    # check the number of unique asins
    assert len(filtered_meta_df) == len(unique_asins)

    if not os.path.exists(f"tmp/{data_abbr[dataset]}_idf_map.pkl"):
        id_f_map = prepare_w2v(filtered_meta_df, train_df)

        with open(f"tmp/{data_abbr[dataset]}_idf_map.pkl", "wb") as f:
            pickle.dump(id_f_map, f)
    else:
        print("Loading id_f_map...")
        id_f_map = pickle.load(open(f"tmp/{data_abbr[dataset]}_idf_map.pkl", "rb"))
        print("Loading id_f_map... done")

    # -------   Remap Feature Matrix -------

    #############
    # Train ids #
    #############
    train_ids = train_df["reviewerID"].tolist() + train_df["asin"].tolist()

    train_ids = sorted(set(train_ids))

    train_id_map = {id_key: index for index, id_key in enumerate(train_ids)}
    rv_train_id_map = {index: id_key for index, id_key in enumerate(train_ids)}

    #############
    # Valid ids #
    #############
    start_index = len(train_id_map)

    val_ids = val_df["reviewerID"].tolist() + val_df["asin"].tolist()

    val_ids = sorted(set(val_ids))
    val_id_map = {
        id_key: index + start_index
        for index, id_key in enumerate(val_ids)
        if id_key not in train_id_map
    }
    rv_val_id_map = {
        index + start_index: id_key
        for index, id_key in enumerate(val_ids)
        if id_key not in train_id_map
    }

    #############
    # Test ids  #
    #############
    start_index = len(train_id_map) + len(val_id_map)

    test_ids = test_df["reviewerID"].tolist() + test_df["asin"].tolist()

    test_ids = sorted(set(test_ids))
    test_id_map = {
        id_key: index + start_index
        for index, id_key in enumerate(test_ids)
        if id_key not in train_id_map and id_key not in val_id_map
    }
    rv_test_id_map = {
        index + start_index: id_key
        for index, id_key in enumerate(test_ids)
        if id_key not in train_id_map and id_key not in val_id_map
    }

    #############
    # Last ids #
    #############
    start_index = len(train_id_map) + len(val_id_map) + len(test_id_map)
    last_ids = ori_review_df["reviewerID"].tolist() + ori_review_df["asin"].tolist()
    last_id_map = {
        id_key: index + start_index
        for index, id_key in enumerate(last_ids)
        if id_key not in train_id_map and id_key not in val_id_map
    }
    rv_last_id_map = {
        index + start_index: id_key
        for index, id_key in enumerate(last_ids)
        if id_key not in train_id_map and id_key not in val_id_map
    }

    review_data = {}

    all_id_map = {**train_id_map, **val_id_map, **test_id_map, **last_id_map}
    rv_all_id_map = {
        **rv_train_id_map,
        **rv_val_id_map,
        **rv_test_id_map,
        **rv_last_id_map,
    }

    with open("tmp/{}_all_id_map.json".format(data_abbr[dataset]), "w") as f:
        json.dump(all_id_map, f)
    with open("tmp/{}_rv_all_id_map.pkl".format(data_abbr[dataset]), "w") as f:
        json.dump(rv_all_id_map, f)


    for id, df in enumerate([train_df, val_df, test_df]):
        # Map the reviewerID and product_id to their unique integer
        s_node = list(map(all_id_map.get, df["reviewerID"]))
        t_node = list(map(all_id_map.get, df["asin"]))

        rating_list = df["overall"].tolist()
        ts_list = df["unixReviewTime"].tolist()

        # Create the edges array without using range
        edge_index = np.array([s_node, t_node], dtype=int)

        # Find unique nodes involved in edges
        uniq_nodes = list(np.unique(edge_index))
        middle_idx = edge_index.shape[1] // 2

        if id == 0:
            review_data["train"] = train_df.to_dict(orient="records")
            # Feature matrix preparation from w2v output
            id_fea_dict = {all_id_map[key]: value for key, value in id_f_map.items()}

            # Sort id_fea_dict by keys
            sorted_id_f_dict = OrderedDict(sorted(id_fea_dict.items()))

            fea_m = []
            for i in uniq_nodes:
                fea = sorted_id_f_dict[i]
                fea_m.append(fea)

            fea_m = np.array(fea_m, dtype=np.float32)

            # Save the feature matrix
            if not args.debug:
                np.save("./tmp/{}_f_m.npy".format(dataset), fea_m)

            # ---------
            new_edge = [s_node + t_node, t_node + s_node]
            new_edge = np.array(new_edge)

            out_edges = []
            for idx, edge in tqdm(enumerate(edge_index.T), total=middle_idx):
                # tmp_edge = str(edge[0]) + '\t' + str(edge[1]) + '\t1'
                tmp_edge = list(map(lambda x: str(x), edge.tolist()))
                tmp_edge = "\t".join(
                    tmp_edge + [f"{rating_list[idx]:.0f}", f"{ts_list[idx]:.0f}"]
                )
                out_edges.append(tmp_edge)

            if not args.debug:
                np.save("./tmp/{}_{}_edge.npy".format(dataset, "train"), new_edge)

                with open(f"tmp/{data_abbr[dataset]}.train.u", "w") as f:
                    f.write("\n".join(out_edges))

        elif id == 1:
            review_data["val"] = val_df.to_dict(orient="records")
            new_edge = [s_node + t_node, t_node + s_node]
            new_edge = np.array(new_edge)

            out_edges = []
            for idx, edge in tqdm(enumerate(edge_index.T), total=middle_idx):
                # tmp_edge = str(edge[0]) + '\t' + str(edge[1]) + '\t1'
                tmp_edge = list(map(lambda x: str(x), edge.tolist()))
                tmp_edge = "\t".join(
                    tmp_edge + [f"{rating_list[idx]:.0f}", f"{ts_list[idx]:.0f}"]
                )
                out_edges.append(tmp_edge)

            if not args.debug:
                np.save("./tmp/{}_{}_edge.npy".format(dataset, "val"), new_edge)

                with open(f"tmp/{data_abbr[dataset]}.val.u", "w") as f:
                    f.write("\n".join(out_edges))

        else:
            # Create test_df's dict
            df_dict = {}
            dup_list = []
            for id, x in enumerate(df[["reviewerID", "asin", "overall","unixReviewTime"]].itertuples()):
                if (
                    all_id_map[x.reviewerID],
                    all_id_map[x.asin],
                    x.overall,
                    x.unixReviewTime
                ) not in df_dict:
                    df_dict[all_id_map[x.reviewerID], all_id_map[x.asin], x.overall, x.unixReviewTime] =  id
                else:
                    dup_list.append((id,df_dict[all_id_map[x.reviewerID], all_id_map[x.asin], x.overall, x.unixReviewTime]))

            user_dict = defaultdict(list)
            rate_dict = defaultdict(list)
            ts_dict = defaultdict(list)

            for idx, edge in tqdm(
                enumerate(edge_index.T[:middle_idx, :]), total=middle_idx
            ):
                tmp_edge = list(map(lambda x: str(x), edge.tolist()))
                u_id = edge[0]
                i_id = edge[1]
                # user_dict[u_id].append(i_id)
                user_dict[u_id].append(i_id)
                rate_dict[u_id].append(rating_list[idx])
                ts_dict[u_id].append(ts_list[idx])

            user_num_items = {u: len(set(user_dict[u])) for u in user_dict}

            # print the average user interaction
            # print(np.mean(list(map(len, user_dict.values()))))

            # Pick the interaction exceed 5-core for each user
            qualified_user_dict = {}

            # Prepare the rating list of qualified users
            qualified_user_rating = {}
            qualified_user_ts = {}

            for u_id in user_dict:
                if user_num_items[u_id] > 5:
                    qualified_user_dict[u_id] = user_dict[u_id]
                    qualified_user_rating[u_id] = rate_dict[u_id]
                    qualified_user_ts[u_id] = ts_dict[u_id]

            # Random select 10 item from qualified_user as test set
            test_out_edges = []
            supp_out_edges = []

            test_s_nodes = []
            test_t_nodes = []

            supp_s_nodes = []
            supp_t_nodes = []

            review_supp_list = []
            review_test_list = []
            np.random.seed(42)
            for u_id, lst_item in qualified_user_dict.items():
                # eliminate the duplicated items
                # items = list(set(lst_item)) # bug here
                items = list(dict.fromkeys(lst_item))
                index_map = {element: lst_item.index(element) for element in items}

                supp_indices = np.random.choice(
                    range(len(items)), min(5, len(items)), replace=False
                )
                supp_items = [lst_item[index_map[items[i]]] for i in supp_indices]
                supp_ratings = [
                    qualified_user_rating[u_id][index_map[items[i]]]
                    for i in supp_indices
                ]
                supp_ts = [
                    qualified_user_ts[u_id][index_map[items[i]]] for i in supp_indices
                ]

                #                   #{total items}         -   #{sup. item}
                remaining_indices = set(range(len(items))) - set(supp_indices)
                remaining_items = [
                    lst_item[index_map[items[i]]] for i in remaining_indices
                ]
                remaining_ratings = [
                    qualified_user_rating[u_id][index_map[items[i]]]
                    for i in remaining_indices
                ]
                remaining_ts = [
                    qualified_user_ts[u_id][index_map[items[i]]]
                    for i in remaining_indices
                ]

                # Ensure the item in the test set is not in the support set
                assert len(set(remaining_items) & set(supp_items)) == 0

                # Create test set
                test_s_nodes += [u_id] * len(remaining_items)
                test_t_nodes += remaining_items

                supp_s_nodes += [u_id] * len(supp_items)
                supp_t_nodes += supp_items

                review_supp_list.extend(
                    [
                        df.iloc[df_dict[(u_id, s_item, s_rating,s_ts)]].to_dict()
                        for s_item, s_rating,s_ts in zip(supp_items, supp_ratings, supp_ts)
                    ]
                )
                review_test_list.extend(
                    [
                        df.iloc[df_dict[(u_id, t_item, t_rating,t_ts)]].to_dict()
                        for t_item, t_rating,t_ts in zip(remaining_items, remaining_ratings,remaining_ts)
                    ]
                )
                supp_out_edges += [
                    str(u_id) + "\t" + str(i_id) + f"\t{r:.0f}" + f"\t{t:.0f}"
                    for i_id, r, t in zip(supp_items, supp_ratings, supp_ts)
                ]

                test_out_edges += [
                    str(u_id) + "\t" + str(i_id) + f"\t{r:.0f}" + f"\t{t:.0f}"
                    for i_id, r, t in zip(
                        remaining_items, remaining_ratings, remaining_ts
                    )
                ]

            support_edge = [supp_s_nodes + supp_t_nodes, supp_t_nodes + supp_s_nodes]

            test_edge = [test_s_nodes + test_t_nodes, test_t_nodes + test_s_nodes]

            # pack support set and test set into list of dict
            review_data["support"] = review_supp_list
            review_data["test"] = review_test_list

            if not args.debug:
                np.save("./tmp/{}_support_edge.npy".format(dataset), support_edge)
                np.save("./tmp/{}_test_edge.npy".format(dataset), test_edge)

                with open(f"tmp/{data_abbr[dataset]}.test.u", "w") as f:
                    f.write("\n".join(test_out_edges))

                with open(f"tmp/{data_abbr[dataset]}.support.u", "w") as f:
                    f.write("\n".join(supp_out_edges))

        # --------- prepare id->text mapping ---------

        unique_asins = df["asin"].unique()
        # only use the asin in df
        filtered_meta_df = meta_df[meta_df["asin"].isin(unique_asins)]

        if id == 0:
            """
                user -> [review1, review2, ...] (train_df)
            """
            user_review_dict = defaultdict(list)
            for reviewer_id, review_text in zip(df["reviewerID"], df["reviewText"]):
                user_review_dict[reviewer_id].append(review_text)

            id_text = {
                all_id_map[key]: " ".join(texts)
                for key, texts in user_review_dict.items()
            }
            item_desc_mapping = dict(
                zip(filtered_meta_df["asin"], filtered_meta_df["description"])
            )

            # id_text.update({ all_id_map[key]: ' '.join(text) for key, text in item_desc_mapping.items()})
            id_text.update(
                {all_id_map[key]: text for key, text in item_desc_mapping.items()}
            )

            json.dump(
                id_text,
                open("./tmp/{}_{}_text.json".format(dataset, "train"), "w"),
                indent=4,
            )

        elif id == 1:
            """
                user -> [review1, review2, ...] (val_df)
            """
            user_review_dict = defaultdict(list)
            for reviewer_id, review_text in zip(df["reviewerID"], df["reviewText"]):
                user_review_dict[reviewer_id].append(review_text)

            id_text = {
                all_id_map[key]: " ".join(texts)
                for key, texts in user_review_dict.items()
            }
            item_desc_mapping = dict(
                zip(filtered_meta_df["asin"], filtered_meta_df["description"])
            )

            id_text.update(
                {all_id_map[key]: text for key, text in item_desc_mapping.items()}
            )

            json.dump(
                id_text,
                open("./tmp/{}_{}_text.json".format(dataset, "val"), "w"),
                indent=4,
            )

        else:
            supp_user_review_dict = defaultdict(list)
            test_user_review_dict = defaultdict(list)

            for reviewer_id, item_id, review_text in zip(
                df["reviewerID"],
                # distribute the item into sup/test
                df["asin"],
                df["reviewText"],
            ):
                if all_id_map[item_id] in supp_t_nodes:
                    supp_user_review_dict[reviewer_id].append(review_text)

                if all_id_map[item_id] in test_t_nodes:
                    test_user_review_dict[reviewer_id].append(review_text)

            supp_id_text = {
                all_id_map[key]: " ".join(texts)
                for key, texts in supp_user_review_dict.items()
            }

            test_id_text = {
                all_id_map[key]: " ".join(texts)
                for key, texts in test_user_review_dict.items()
            }

            item_desc_mapping = dict(
                zip(filtered_meta_df["asin"], filtered_meta_df["description"])
            )

            supp_id_text.update(
                {all_id_map[key]: text for key, text in item_desc_mapping.items()}
            )
            test_id_text.update(
                {all_id_map[key]: text for key, text in item_desc_mapping.items()}
            )

            json.dump(
                supp_id_text,
                open(
                    "./tmp/{}_support_text.json".format(dataset),
                    "w",
                ),
                indent=4,
            )
            json.dump(
                test_id_text,
                open("./tmp/{}_test_text.json".format(dataset), "w"),
                indent=4,
            )
    ###############################################

    old_user_items = defaultdict(list)

    sup_test_user = set()
    for _type in ["train", "val", "support", "test"]:
        for data in review_data[_type]:
            user = data["reviewerID"]
            user_id = str(all_id_map[user])
            item = data["asin"]
            item_id = str(all_id_map[item])
            
            # if _type == 'support' or _type == 'test':
            if _type == 'test':
                if user_id not in sup_test_user:
                    sup_test_user.add(user_id)
            old_user_items[user_id].append(item_id)


    # Only keep 5-core user
    user_items = {}
    for key, value in old_user_items.items():
        if key in sup_test_user:
            continue
            user_items[key] = value
        else:
            if len(value) >= 5:
                user_items[key] = value


    #user_items = {key: value for key, value in old_user_items.items() if len(value) >= 5}
    def belong_to(id):
        train_len = len(review_data["train"])
        val_len = len(review_data["val"])
        support_len = len(review_data["support"])
        test_len = len(review_data["test"])

        if id < train_len:
            return "train", id
        elif id >= train_len and id < train_len + val_len:
            return "val", id - train_len
        elif id >= train_len + val_len and id < train_len + val_len + support_len:
            return "support", id - train_len - val_len
        else:
            return "test", id - train_len - val_len - support_len

    user_id2name = {}
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}

    test_user_id2name = {}
    test_user2id = {}
    test_item2id = {}
    test_id2user = {}
    test_id2item = {}

    id_be_removed = defaultdict(list)
    id_be_added = defaultdict(list)
    all_review_data = (
        review_data["train"]
        + review_data["val"]
        + review_data["support"]
        + review_data["test"]
    )
    for i in trange(len(all_review_data), total=len(all_review_data)):
        skip_flag = False
        user = all_review_data[i]["reviewerID"]
        user_id = str(all_id_map[user])
        item = all_review_data[i]["asin"]
        item_id = str(all_id_map[item])

        if user_id in user_items:
            key, id = belong_to(i)
            id_be_added[key].append(id)
        else:
            key, id = belong_to(i)

            # only remove (< 5-core) user in train and val
            # if key == 'train' or key == 'val':
            if key != 'test':
                id_be_removed[key].append(id)
                skip_flag = True
            else:
                id_be_added[key].append(id)

            if "reviewerName" in all_review_data[i]:
                test_user_id2name[user_id] = all_review_data[i]["reviewerName"]
            else:
                test_user_id2name[user_id] = all_review_data[i]["reviewerID"]
            test_user2id[user] = user_id
            test_id2user[user_id] = user
            test_item2id[item] = item_id
            test_id2item[item_id] = item
            continue

        # if skip_flag:
        #     continue


        if "reviewerName" in all_review_data[i]:
            user_id2name[user_id] = all_review_data[i]["reviewerName"]
        else:
            user_id2name[user_id] = all_review_data[i]["reviewerID"]

        user2id[user] = user_id
        id2user[user_id] = user
        item2id[item] = item_id
        id2item[item_id] = item

    breakpoint()
    assert len(user_id2name) == len(user2id)
    assert len(user_items) == len(user2id)

    ###################################################################
    new_review_data = {}

    iter_keys = id_be_removed.keys() if len(id_be_removed) > len(id_be_added) else id_be_added.keys()

    for key in tqdm(iter_keys, total=len(iter_keys)):

        if len(id_be_removed[key]) == 0 :
            new_review_data[key] = review_data[key]
            continue

        if len(id_be_removed[key]) > len(id_be_added[key]) and len(id_be_added[key]) > 0:

            ids = id_be_removed[key]
            tmp_list = []
            for i, data in tqdm(enumerate(review_data[key]), total=len(review_data[key])):
                if i not in ids:
                    tmp_list.append(data)
            new_review_data[key] = tmp_list
        else:
            ids = id_be_added[key]
            tmp_list = []
            for i, data in tqdm(enumerate(review_data[key]), total=len(review_data[key])):
                if i in ids:
                    tmp_list.append(data)
            new_review_data[key] = tmp_list

    # Replace the old review_data with the new one
    for key in id_be_removed:
        review_data[key] = new_review_data[key]
        

    ###################################################################

    with open(f"tmp/{data_abbr[dataset]}_sequential.txt", "w") as out:
        for user, items in user_items.items():
            out.write(user + " " + " ".join(items) + "\n")

    sample_test_data(dataset, f"tmp/{data_abbr[dataset]}_sequential.txt")

    ###################################################################


    # dump user2id, item2id, id2user, id2item
    test_datampas = {
        "user2id": test_user2id,
        "item2id": test_item2id,
        "id2user": test_id2user,
        "id2item": test_id2item,
    }
    datamaps = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    }

    datamaps = extract_from_df(qualified_review_df, cat_dict, datamaps)
    test_datampas = extract_from_df(qualified_review_df, cat_dict, test_datampas)


    json.dump(
        datamaps,
        open(f"./tmp/{data_abbr[dataset]}_datamaps.json", "w"),
        indent=4,
    )
    with open(f"tmp/{data_abbr[dataset]}_user_id2name.pkl", "wb") as f:
        pickle.dump(user_id2name, f, protocol=pickle.HIGHEST_PROTOCOL)

    json.dump(
        test_datampas,
        open(f"./tmp/{data_abbr[dataset]}_test_datamaps.json", "w"),
        indent=4,
    )

    with open(f"tmp/{data_abbr[dataset]}_test_user_id2name.pkl", "wb") as f:
        pickle.dump(test_user_id2name, f, protocol=pickle.HIGHEST_PROTOCOL)

    all_review_data = (
        review_data["train"]
        + review_data["val"]
        + review_data["support"]
        + review_data["test"]
    )

    with open(f"tmp/{data_abbr[dataset]}_review_splits.pkl", "wb") as f:
        pickle.dump(review_data, f, protocol=pickle.HIGHEST_PROTOCOL)



    ## check

    for r_data in all_review_data:
        user = r_data["reviewerID"]
        item = r_data["asin"]

        if user not in user2id:
            print("something wrong with user2id")
            import ipdb

            ipdb.set_trace()
        if item not in item2id:
            print("something wrong with item2id")
            import ipdb

            ipdb.set_trace()

    print("done !")
    breakpoint()


# ----------------------------


# Main function calls
# name = "Musical_Instruments"
name = args.dataset
file = f"./data/{name}.json.gz"
meta_file = f"./data/meta_{name}.json.gz"

get_data(file, meta_file, name)
