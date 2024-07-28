import pickle
import os
import sys
import traceback
import numpy as np
from tqdm import trange, tqdm
import torch
import pandas as pd
from torch.optim.lr_scheduler import StepLR, ExponentialLR
0
from loguru import logger

logger_format = "<green>[{time:MM/DD-HH:mm:ss}]</green> {message}"
logger.configure(handlers=[{"sink": sys.stderr, "format": logger_format}])


try:
    # Data associated
    from pysmore.input import reader

    # Data Loader
    from pysmore.data.DataGenerator import (
        RetrievalDataGenerator,
        RankingDataGenerator,
        ParallelSessDataGenerator,
    )

    # Trainer
    from pysmore.trainer.RetrievalTrainer import TripletTrainer
    from pysmore.trainer.RankingTrainer import FeatureWisedTrainer
    from pysmore.trainer.TrainerArgs import TrainingArgs
    from pysmore.trainer.evaluate import evaluate

    # Utils
    from pysmore.utils.config import proc_args
    import pysmore.utils.utils as utils
    from pysmore.models.lr import LRPolicyScheduler

    from pysmore.models.CLIP import CLIP
except ModuleNotFoundError:
    # Data associated
    from input import reader

    # Data Loader
    from data.DataGenerator import (
        RetrievalDataGenerator,
        RankingDataGenerator,
        ParallelSessDataGenerator,
    )

    # Trainer
    from trainer.RetrievalTrainer import TripletTrainer
    from trainer.RankingTrainer import FeatureWisedTrainer
    from trainer.TrainerArgs import TrainingArgs
    from trainer.evaluate import evaluate

    # Utils
    from utils.config import proc_args
    import utils.utils as utils
    from models.lr import LRPolicyScheduler

    from models.CLIP import CLIP


class CLIP_Config:
    def __init__(self, args):
        args_dict = {
            "aggregation_times": 2,
            "ft_epoch": 50,
            "lr": 2e-05,
            "batch_size": 64,
            "gnn_input": 128,
            "gnn_hid": 128,
            "gnn_output": 128,
            "edge_coef": 0.1,
            "neigh_num": 3,
            "num_labels": 5,
            "k_spt": 5,
            "k_val": 5,
            "k_qry": 50,
            "n_way": 5,
            "context_length": 128,
            "coop_n_ctx": args.n_ctx,
            "prompt_lr": 0.01,
            "position": "end",
            "class_specific": False,
            "ctx_init": True,
            "embed_dim": 128,
            "transformer_heads": 8,
            "transformer_layers": 12,
            "transformer_width": 512,
            "vocab_size": 49408,
            "gpu": 0,
        }
        for k, v in args_dict.items():
            setattr(self, k, v)


def main(args, hyper_trial=None):
    if args.silence:
        logger.remove()
        logger.add(sys.stderr, format=logger_format, level="WARNING")

    # 0. Initialize seed
    torch.multiprocessing.set_sharing_strategy("file_system")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Parse  Input
    # ----------------
    # * Parse the input format and annotate it with defined names
    # - e.g. input format: ["c1-c2", "y"]
    # parsed format: ["u@c1", "u@c2", "y"]
    (
        col_names,  # ['F@c1', 'F@u', 'y'] or ['u', 'i', 'r']
        col_types,  # ['u.c', 'u.c', 'y']
        user_cols,  # For retrieval task
        item_cols,  # For retrieval task
        target_cols,  # Both retrieval and ranking task need
        time_cols,  # Some dataset has time information
        sparse_cols,  # For ranking task, specifiy the sparse features
        dense_cols,  # For ranking task, specifiy the dense features
        seq_cols,  # For ranking task, specifiy the seq features
    ) = utils.ParseInput(args.format, args.task)

    # sparse_cols = list(filter(lambda x:len(x)>0, sparse_cols))
    # dense_cols  = list(filter(lambda x:len(x)>0, dense_cols))
    # seq_cols    = list(filter(lambda x:len(x)>0, seq_cols))

    tokenizer = None

    # 2. Read Data
    ##

    logger.info(f"Reading Train file: {args.train}")

    dtype_dict = dict(zip(col_names, col_types))
    dtype_dict = {k: ("float" if "d" in v else "str")
                  for k, v in dtype_dict.items()}

    utils._check_input_format(args.format, args.train, args.sep)
    train_df = pd.read_csv(
        args.train,
        sep=args.sep,
        header=0 if args.header else None,
        names=col_names,
        engine="python" if len(args.sep) > 0 else "c",
        dtype=dtype_dict,
    )
    logger.info(f"Reading Train file: {args.train} ... done ({len(train_df)})")

    logger.info("Glimpse of Train Data: ")
    if not args.silence:
        print(train_df.head())

    # 3. Feature Preprocessing & Encoding
    ##

    logger.info("Feature Preprocessing & Encoding... ")
    sp_vocab_size = []
    (
        train_df,
        n_classes,  # Uniq Target number
        # feat_dims_size,  # Every Sparse feature dimension
        lb_enc_list,  # Reserve the same lb_enc used for val set (Sparse)
        vocab_list,  # Reverse the same vocab used for val set (Seq)
        user_map,  # Reverse the same userID_map used for val set
        item_map,  # Reverse the same itemID_map used for val set
        rv_user_map,
        rv_item_map,
    ) = utils.FeaturePreprocessing(
        train_df,
        target_cols,
        sparse_cols,
        dense_cols,
        seq_cols,
        user_cols,
        item_cols,
        enc_policy=args.sp_enc_policy,
        sp_vocab_size=sp_vocab_size,
        tokenizer=tokenizer,
    )

    if args.CLIP_path:
        import json

        with open(args.text_json) as f:
            tit_dict = json.load(f)

            node2texts = {}
            for key, text in tit_dict.items():
                node2texts[key] = text

        clip_cfg = CLIP_Config(args)

        if hyper_trial:
            clip_cfg.coop_n_ctx = hyper_trial.suggest_int("n_ctx", 1, 127)

        clip_model = CLIP(clip_cfg)
        data_abbr = {
            "Musical_Instruments": "MI",
            "All_Beauty": "BE",
            "Industrial_and_Scientific": "IS",
            "Sports_and_Outdoors": "SO",
            "Toys_and_Games": "TG",
            "Arts_Crafts_and_Sewing": "AC",
            "reviews_Beauty_5": "BE5"
        }
        data_name = data_abbr.get(args.CLIP_path.split("/")[1], "MI")

        if not torch.cuda.is_available():
            clip_model.load_state_dict(torch.load(args.CLIP_path, map_location=torch.device('cpu')))
        else:
            clip_model.load_state_dict(torch.load(args.CLIP_path))

        if not os.path.exists(f"./{data_name}_id_{args.text_by_idx}_text.pkl"):
            id_texts = {}
            _idx = args.text_by_idx
            for k in tqdm(node2texts, desc="Construct texts map"):
                try:
                    id_texts[lb_enc_list[_idx].transform(
                        [k])[_idx]] = node2texts[k]
                except:
                    continue
            sorted(list(id_texts.keys()))

            with open(f"./{data_name}_id_{args.text_by_idx}_text.pkl", "wb") as f:
                pickle.dump(id_texts, f)
        else:
            with open(f"./{data_name}_id_{args.text_by_idx}_text.pkl", "rb") as f:
                id_texts = pickle.load(f)
            id_texts = [id_texts[id] for id in sorted(id_texts.keys())]

    if args.val:
        logger.info(f"Reading Validation file: {args.val}")

        utils._check_input_format(args.format, args.val, args.sep)
        val_df = pd.read_csv(
            args.val,
            sep=args.sep,
            header=0 if args.header else None,
            names=col_names,
            engine="python" if len(args.sep) > 0 else "c",
            dtype=dtype_dict,
        )

        logger.info(
            f"Reading Validation file: {args.val} ... done ({len(val_df)})")

        # 3. Feature Preprocessing (Optional)
        val_df, *_ = utils.FeaturePreprocessing(
            val_df,
            target_cols,
            sparse_cols,
            dense_cols,
            seq_cols,
            user_cols,
            item_cols,
            enc_policy=args.sp_enc_policy,
            sp_vocab_size=sp_vocab_size,
            enc_list=lb_enc_list,
            vocab_list=vocab_list,
            tokenizer=tokenizer,
        )
    if args.test:
        logger.info(f"Reading Validation file: {args.test}")

        utils._check_input_format(args.format, args.test, args.sep)
        test_df = pd.read_csv(
            args.test,
            sep=args.sep,
            header=0 if args.header else None,
            names=col_names,
            engine="python" if len(args.sep) > 0 else "c",
            dtype=dtype_dict,
        )

        logger.info(
            f"Reading Test file: {args.test} ... done ({len(test_df)})")

        # 3. Feature Preprocessing (Optional)
        test_df, *_ = utils.FeaturePreprocessing(
            test_df,
            target_cols,
            sparse_cols,
            dense_cols,
            seq_cols,
            user_cols,
            item_cols,
            enc_policy=args.sp_enc_policy,
            sp_vocab_size=sp_vocab_size,
            enc_list=lb_enc_list,
            vocab_list=vocab_list,
            tokenizer=tokenizer,
        )

    # ------------- Finish Preprocessing -------------

    (user_features, item_features, user_meta_cols, item_meta_cols) = utils.ParseFeature(
        input_names=col_names,
        input_types=col_types,
        sp_vocab_size=sp_vocab_size,
        neg_sample=(args.task != "rank"),
        embed_dim=args.embed_dim,
        pooling_opt=args.pooling_opt,
        ignore_feat=[time_cols, target_cols],
    )

    # ----------- Load Pretrain Embedding ------------

    pretrain_embs = utils.LoadPretrain(
        logger, args, sp_vocab_size, lb_enc_list, col_names
    )

    # ------------- Generate Dataset -----------------
    if args.task == "sess":
        dg = ParallelSessDataGenerator(
            # only accept UIR-df
            train_df,
            user_cols,
            item_cols,
            time_cols,
            args.batch_size,
            args.sampler,
            args.num_neg,
        )

        (train_dl) = dg.generate_data()

    elif args.task == "retrieval":
        dg = RetrievalDataGenerator(
            # only accept UIR-df
            train_df[[user_cols, item_cols, target_cols]],
            user_cols,
            item_cols,
            args.sampler,
            args.num_neg,
        )

        user_meta_df, item_meta_df = None, None

        if len(user_meta_cols) > 0:
            user_meta_df = (
                train_df[[user_cols] + user_meta_cols]
                .drop_duplicates()
                .set_index(user_cols, drop=False)
            )

        if len(item_meta_cols) > 0:
            item_meta_df = (
                train_df[[item_cols] + item_meta_cols]
                .drop_duplicates()
                .set_index(item_cols, drop=False)
            )

        (
            train_dl,
            user_dl,
            item_dl,
        ) = dg.generate_data(args.batch_size, args.worker, user_meta_df, item_meta_df)

    elif args.task == "rank":
        train_y = train_df[target_cols]
        if len(time_cols):
            train_t = train_df[time_cols]
            del train_df[time_cols]

        # del train_df[target_cols]
        train_x = train_df.drop(target_cols, axis=1)

        if args.val:
            val_y = val_df[target_cols]

            if len(time_cols):
                val_t = val_df[time_cols]
                del val_df[time_cols]

            # del val_df[target_cols]
            val_x = val_df.drop(target_cols, axis=1)
        else:
            val_x = None
            val_y = None

        if args.test:
            test_y = test_df[target_cols]

            if len(time_cols):
                test_t = test_df[time_cols]
                del test_df[time_cols]

            # del val_df[target_cols]
            test_x = test_df.drop(target_cols, axis=1)
        else:
            test_x = None
            test_y = None

        dg = RankingDataGenerator(train_x, train_y)

        (train_dl, train_data, val_dl, val_data, test_dl, test_data) = dg.generate_data(
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            split_ratio=[args.split_ratio],
            split_strategy="random",
            batch_size=args.batch_size,
            num_workers=args.worker,
            contain_sequence=len(seq_cols) > 0,
            context_length=args.context_length,
        )

    # --------------- Finish Genrate Data -----------

    device = torch.device(args.device)

    # use torch.nn loss function
    default_loss = True
    try:
        loss_cls = utils.dynamic_import_cls("torch.nn", f"{args.loss_fn}Loss")
    except:
        default_loss = False
        pass
    if not default_loss:
        try:
            loss_cls = utils.dynamic_import_cls(
                f"models.loss", f"{args.loss_fn}")
        except:
            loss_cls = utils.dynamic_import_cls(
                f"pysmore.models.loss", f"{args.loss_fn}"
            )

    try:
        model_cls = utils.dynamic_import_cls(
            f"pysmore.models.{args.model}", f"{args.model}"
        )
    except:
        model_cls = utils.dynamic_import_cls(
            f"models.{args.model}", f"{args.model}")
    if args.metrics is not None:
        try:
            metric_fns = [
                utils.dynamic_import_cls("pysmore.trainer.metric", metric)
                for metric in args.metrics
            ]
        except:
            metric_fns = [
                utils.dynamic_import_cls("trainer.metric", metric)
                for metric in args.metrics
            ]
    else:
        metric_fns = None

    # Model initialization
    try:
        tmp_dict = {}
        if args.CLIP_path:
            tmp_dict = {
                "id_texts": id_texts,
                "clip_model": clip_model,
                "clip_cfg": clip_cfg,
                "item_map": item_map,
                "rv_item_map": rv_item_map,
            }

        model = model_cls(
            features=user_features + item_features,
            mlp_params=args.mlp_params,
            pretrain_embs=pretrain_embs,
            inter_df=train_df[[user_cols, item_cols, target_cols]],
            n_classes=n_classes,
            context_length=args.context_length,
            koint_output=(args.task == "rank"),
            device=device,
            embed_init=args.embed_init,
            **tmp_dict,
        )
        model = model.to(device)

    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(exc_type)
        print(exc_value)
        traceback.print_tb(exc_traceback)
        exit(1)

    if hyper_trial is not None:
        """
        args.embed_init = hyper_trial.suggest_categorical(
            "embed_init", ["rand", "smore", "xariver_normal",
                           "xariver_uniform", "kaiming_normal", "kaiming_uniform"])
        """

        optimizer_name = hyper_trial.suggest_categorical(
            "optim", ["Adam", "RMSprop", "SGD"]
        )
        lr = hyper_trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        optim_cls = utils.dynamic_import_cls(
            "torch.optim", f"{optimizer_name}")

        weight_decay = hyper_trial.suggest_float(
            "weight_decay", 1e-5, 1e-1, log=True)

        optimizer = optim_cls(model.parameters(), lr=lr,
                              weight_decay=weight_decay)

        args.max_epochs = hyper_trial.suggest_int("max_epochs", 10, 100)
        args.batch_size = hyper_trial.suggest_int("batch_size", 4, 256)

    else:
        optim_cls = utils.dynamic_import_cls("torch.optim", f"{args.optim}")

        optimizer = optim_cls(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    loss_cls = loss_cls()
    # epoch should be the graph update whole interactions.
    # update_times = int(args.max_epochs * 1000000 * args.num_neg)
    update_times = int(args.max_epochs * 1000 * args.num_neg)

    scheduler = LRPolicyScheduler(
        optimizer,
        base_lr=args.lr,
        decay_interval=10000,
        update_steps=args.batch_size,
        total_steps=update_times,
    )
    # scheduler = StepLR(
    #     optimizer, 1.0, gamma=0.1) if args.task == 'rank' else scheduler

    scheduler = ExponentialLR(
        optimizer, 0.99) if args.task == "rank" else scheduler

    scheduler = None

    if not args.silence:
        utils.print_status(
            args,
            model,
            len(sparse_cols) > 0,
            len(seq_cols) > 0,
            len(dense_cols) > 0,
        )
    # --------------- Create Trainer & fit -----------

    if args.task == "sess":
        val_dl = None
        trainer = ParallelSessRecTrainer(
            device,
            model,
            loss_cls,
            optimizer,
            scheduler,
            train_dl,
            val_dl,
            metric_fns=metric_fns,
            tokernizer=tokenizer,
            data_collator=None,
            optuna_trial=hyper_trial,
        )
        trainer.fit()

    elif args.task == "retrieval":
        trainer = TripletTrainer(
            device,
            model,
            args.max_epochs,
            loss_cls,
            update_times,
            optimizer,
            optimizer_params={
                "lr": args.lr,
            },
            lr_scheduler=scheduler,
        )

    elif args.task == "rank":
        training_args = TrainingArgs(
            output_dir=args.saved_path,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            n_train_epochs=args.max_epochs,
            n_targets=n_classes,
            weight_decay=args.weight_decay,
            learning_rate=args.lr,
            es_patience=args.es_patience,
            es_by=args.es_by,
            log_interval=args.log_interval,
            silence=args.silence,
        )

        trainer = FeatureWisedTrainer(
            training_args,
            device,
            model,
            loss_cls,
            optimizer,
            scheduler,
            train_dl,
            val_dl,
            metric_fns=metric_fns,
            optuna_trial=hyper_trial,
        )

    if args.task == "retrieval":
        model, x = trainer.fit(train_dl)

        if args.saved_option == "embedding":
            if "prompt" in args.model:
                user_emb, item_emb = model.get_embedding()
            else:
                user_emb, item_emb = trainer.extract_embedding(
                    [user_cols, item_cols])

            utils.save_embedding(
                user_emb, item_emb, rv_user_map, rv_item_map, args.saved_path
            )

        else:  # saved model
            print("\nSaved Model into: {}".format(args.saved_path))
            torch.jit.save(torch.jit.trace(model, (x)), args.saved_path)

    elif args.task == "rank":
        model, x = trainer.fit()
        # torch.onnx.export(
        # model,
        # {"x": x},
        # f="exp/dcn.ckpt.onnx",
        # output_names=['output'],
        # dynamic_axes={"x": {0: "batch_size"}, 'output': {0: 'batch_size'}},
        # )

        # print("\nSaved Model into: {}".format(args.saved_path))
        # torch.jit.save(torch.jit.trace(model, (x)), args.saved_path)
        # model = torch.jit.load(args.saved_path)
        model = torch.load(args.saved_path)
        if test_dl is not None:
            loss, metric_dict, _ = evaluate(
                metric_fns, model, device, loss_cls, test_dl, debug=True
            )

            output = []
            output.append(",".join(args.metrics))
            output.append(",".join(map(str, metric_dict.values())))

            if not args.silence:
                print("========= Test Result ======= ")
                print("\n".join(output))
        else:
            loss, metric_dict, _ = evaluate(
                metric_fns, model, device, loss_cls, val_dl)
            if not args.silence:
                print("========= Validation Result ======= ")
                print(metric_dict)

        if "loss" in args.es_by:
            return loss
        else:
            return list(metric_dict.values())[0]


# --------------------------------


def entry_points():
    args = proc_args()

    if args.hyper_opt:
        import optuna

        args.silence = True
        opt_direction = "maximize" if args.es_by[-1] == "+" else "minimize"
        study = optuna.create_study(direction=opt_direction)
        study.optimize(
            lambda trial: main(args, trial), n_trials=100, show_progress_bar=True
        )
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            if key in args.__dict__:
                setattr(args, key, value)
        # Final Runs again
        args.silence = False
        main(args)
        print(
            f"Best Trial Value: {best_trial.value} ({opt_direction})({args.es_by})")
        print("Best Trial Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        print("Importance of hyperparameters:")
        importance = optuna.importance.get_param_importances(study)
        for param, importance in importance.items():
            print(f"    {param}: {importance}")

    else:
        main(args)


if __name__ == "__main__":
    entry_points()
