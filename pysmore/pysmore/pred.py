import sys
import numpy as np
from tqdm import trange, tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torchtext.data.utils import get_tokenizer
import onnxruntime
import onnx

try:
    # Data associated
    from input import reader
    # Data Loader
    from data.DataGenerator import (RetrievalDataGenerator,
                                    RankingDataGenerator,
                                    DLRMDataGenerator,
                                    PandasDataset)
    # Utils
    from utils.test_config import proc_args
    import utils.utils as utils
    from models.lr import LRPolicyScheduler
except:
    # Data associated
    from pysmore.input import reader
    # Data Loader
    from pysmore.data.DataGenerator import (RetrievalDataGenerator,
                                            RankingDataGenerator,
                                            DLRMDataGenerator,
                                            PandasDataset)
    # Utils
    from pysmore.utils.test_config import proc_args
    import pysmore.utils.utils as utils
    from pysmore.models.lr import LRPolicyScheduler


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float64)


def inference(model, device, data_loader=None, observed_dict=None, user_map=None, rv_user_map=None, item_map=None, rv_item_map=None):
    model = model.to(device)
    model.eval()
    query_pred = {}
    with torch.no_grad():
        for id, x_dict in enumerate(tqdm(data_loader)):
            fea = {k: v.to(device) for k, v in x_dict.items()}
            try:
                observed_items = observed_dict[fea['u'][0].item()]
            except:
                observed_items = []

            scores = model(fea)
            topk = torch.topk(scores, 1000).indices
            query_pred[x_dict['u'][0].item()] = [t for t in topk.tolist()
                                                 if t not in observed_items][:1000]

    return query_pred


def eval(model, test_dl, device, metric_fns):

    def _eval(predictions, answers, verbose=False):

        metric_dict = {}
        for metric in metric_fns:
            metric_name = metric.__name__
            metric_val = metric(predictions, answers)

            if verbose:
                print(
                    "| {} {:8.3f} | ".format(
                        metric_name, metric_val
                    )
                )
            metric_dict[metric_name] = metric_val
        return metric_dict

    pbar = test_dl
    predictions, answers = [], []

    # ort_session = onnxruntime.InferenceSession("exp/dcn.ckpt.onnx", providers=["CPUExecutionProvider"])

    # model.eval()
    with torch.no_grad():
        for x_dict, targets in pbar:
            features = {k: v.to(device) for k, v in x_dict.items()}

            targets = targets.to(device)

            # rt_inputs= {ort_session.get_inputs()[i].name: to_numpy(v) for i, (k,v) in enumerate(features.items())}
            # preds = ort_session.run(None, rt_inputs)[0]

            preds = model(features)

            predictions += preds.argmax(1).tolist() \
                if len(preds.shape) > 1 else preds.tolist()

            answers += targets.tolist()

    predictions = torch.tensor(predictions)
    answers = torch.tensor(answers)
    metric_dict = _eval(predictions, answers)

    return metric_dict


def predict_data_generator(test_df, target_cols):
    test_y = test_df[target_cols]
    del test_df[target_cols]
    test_x = test_df

    test_dataset = PandasDataset(test_x, test_y)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.worker,
                             )

    return test_dataset, test_loader


def rank_data_generator(model):
    observed_dict = {}

    for row in train_df.itertuples():
        if row.u not in observed_dict:
            observed_dict[row.u] = [row.i]
        else:
            observed_dict[row.u].append(row.i)

    result = trainer.inference(
        model, inter_dl, observed_dict, user_map, rv_user_map, item_map, rv_item_map)

    print("write to answer")
    output = []
    for uid in tqdm(ans_dict):
        if uid not in result:
            continue

        ui_results = [str(rv_user_map[uid]), str(len(ans_dict[uid]))]

        for idx, rid in enumerate(result[uid], 1):
            if idx > len(ans_dict[uid]):
                break
            if rid in ans_dict[uid]:
                ui_results.append('1')
            else:
                ui_results.append('0')
        output.append(' '.join(ui_results))

    print(args.saved_path+".ui.rec")
    with open(args.saved_path+".ui.rec", 'w') as f:
        f.writelines("\n".join(output))


def main(args):

    # 0. Initialize seed
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # 1. Parse  Input
    # ----------------
    # * Parse the input format and annotate it with defined names
    # - e.g. input format: ["c1-c2", "y"]
    # parsed format: ["u@c1", "u@c2", "y"]

    (
        col_names,  # ['u@c1', 'u@c2', 'y'] or ['u', 'i', 'r']
        col_types,  # ['u.c', 'u.c', 'y']
        user_cols,  # For retrieval task
        item_cols,  # For retrieval task
        target_cols,  # Both retrieval and ranking task need
        time_cols,  # Some dataset has time information
        sparse_cols,  # For ranking task, specifiy the sparse features
        dense_cols,  # For ranking task, specifiy the dense features
        seq_cols  # For ranking task, specifiy the seq features
    ) = utils.ParseInput(args.format, args.task)

    sparse_cols = list(filter(lambda x: len(x) > 0, sparse_cols))
    dense_cols = list(filter(lambda x: len(x) > 0, dense_cols))
    seq_cols = list(filter(lambda x: len(x) > 0, seq_cols))

    tokenizer = get_tokenizer("basic_english") \
        if len(seq_cols) > 0 else None

    # 2. Read Data
    print(f"Reading Train file: \n\t{args.train}", end=' ')
    sys.stdout.flush()

    dtype_dict = dict(zip(col_names, col_types))
    dtype_dict = {k: ("float" if 'd' in v else "str")
                  for k, v in dtype_dict.items()}

    train_df = pd.read_csv(args.train,
                           sep=args.sep,
                           header=0 if args.header else None,
                           names=col_names,
                           dtype=dtype_dict)

    print(f"... done ({len(train_df)})")

    # 3. Feature Preprocessing (Optional)
    print("Feature Preprocessing ", end=' ')
    sys.stdout.flush()

    (train_df,
     n_classes,
     feat_dims_size,
     lb_enc_list,
     vocab_list,
     user_map,
     item_map,
     rv_user_map,
     rv_item_map) = utils.FeaturePreprocessing(train_df,
                                               target_cols,
                                               sparse_cols,
                                               dense_cols,
                                               seq_cols,
                                               user_cols,
                                               item_cols,
                                               tokenizer=tokenizer)

    (user_features,
     item_features,
     user_meta_cols,
     item_meta_cols) = utils.ParseFeature(col_names,
                                          col_types,
                                          feat_dims_size,
                                          (args.task != 'rank'),
                                          embed_dim=args.embed_dim if args.embed_dim > 0
                                          else None)  # embed_dim=None -> auto

    # Reading Test Data
    print(f"Reading Test file: \n\t{args.test}", end=' ')
    sys.stdout.flush()

    test_df = pd.read_csv(args.test,
                          sep=args.sep,
                          header=0 if args.header else None,
                          names=col_names,
                          dtype=dtype_dict)

    print(f"... done ({len(test_df)})")

    # 3. Feature Preprocessing (Optional)
    test_df, *_ = utils.FeaturePreprocessing(test_df,
                                             target_cols,
                                             sparse_cols,
                                             dense_cols,
                                             seq_cols,
                                             user_cols,
                                             item_cols,
                                             lb_enc_list,
                                             vocab_list,
                                             tokenizer=tokenizer)

    # ----------
    # model = onnx.load("exp/dcn.ckpt.onnx")
    # onnx.checker.check_model(model)

    print("Loading Model from ", args.ckpt_path)
    model = torch.jit.load(args.ckpt_path)

    if args.task == 'retrieval':

        dg = DLRMDataGenerator(
            # only accept UIR-df
            train_df[[user_cols, item_cols, target_cols]],
            test_df,
            user_cols,
            item_cols)

        # user_meta_df, item_meta_df = None, None

        user_meta_df = train_df[[
            user_cols]+user_meta_cols].drop_duplicates().set_index(user_cols, drop=False)

        item_meta_df = train_df[[
            item_cols]+item_meta_cols].drop_duplicates().set_index(item_cols, drop=False)

        train_dl, inter_dl, ans_dict = dg.generate_data(args.batch_size,
                                                        args.worker,
                                                        user_meta_df,
                                                        item_meta_df)

        observed_dict = {}

        for row in train_df.itertuples():
            if row.u not in observed_dict:
                observed_dict[row.u] = [row.i]
            else:
                observed_dict[row.u].append(row.i)

        result = inference(model, args.device, inter_dl, observed_dict,
                           user_map, rv_user_map, item_map, rv_item_map)

        print("write to answer")
        output = []
        for uid in tqdm(ans_dict):
            if uid not in result:
                continue

            ui_results = [str(rv_user_map[uid]), str(len(ans_dict[uid]))]

            for idx, rid in enumerate(result[uid], 1):
                if idx > len(ans_dict[uid]):
                    break
                if rid in ans_dict[uid]:
                    ui_results.append('1')
                else:
                    ui_results.append('0')
            output.append(' '.join(ui_results))

        print(args.saved_path+".ui.rec")
        with open(args.saved_path+".ui.rec", 'w') as f:
            f.writelines("\n".join(output))
    else:
        train_y = train_df[target_cols]
        del train_df[target_cols]
        train_x = train_df

        test_y = test_df[target_cols]
        del test_df[target_cols]
        test_x = test_df
        # test_dataset = PandasDataset(test_x, test_y)

        # test_dl = DataLoader(test_dataset,
        # batch_size=args.batch_size,
        # shuffle=False,
        # num_workers=args.worker,
        # )

        dg = RankingDataGenerator(train_x, train_y)

        (train_dl, train_data,
         test_dl, test_data,) = dg.generate_data(
            val_x=test_x, val_y=test_y,
            batch_size=args.batch_size,
            num_workers=args.worker,
            contain_sequence=len(seq_cols) > 0)
        if args.metrics is not None:
            try:
                metric_fns = [utils.dynamic_import_cls(
                    "pysmore.trainer.metric", metric) for metric in args.metrics]
            except:
                metric_fns = [utils.dynamic_import_cls(
                    "trainer.metric", metric) for metric in args.metrics]
        else:
            metric_fns = []

        score = eval(model, test_dl, args.device, metric_fns)
        print(score)

        output = []
        output.append(",".join(args.metrics))
        output.append(",".join(map(str, score.values())))

        print("Output to ", args.saved_path+".csv")
        with open(args.saved_path+".csv", 'w') as f:
            f.write("\n".join(output))


def entry_points():
    args = proc_args()
    # dataset = pre_process(args)
    main(args)


if __name__ == '__main__':
    entry_points()
