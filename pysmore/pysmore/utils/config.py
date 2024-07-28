import sys
import os
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def dash_separated_strs(value):
    try:
        vals = value.split("-")
    except ValueError:
        raise argparse.ArgumentTypeError(
            "%s is not a valid dash separated list of ints" % value
        )

    return vals


def setup_by_args(args):
    pass


def proc_args():
    try:
        from pysmore.trainer import metric
    except:
        from trainer import metric

    # Parse argument
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help="Select Model")
    parser.add_argument('--loss_fn',
                        type=str,
                        required=True,
                        help="Select Loss Function")

    parser.add_argument('--embed_dim',
                        type=int,
                        default=0,
                        help="Dimension for embedding")

    parser.add_argument('--mlp_params',
                        action=type('', (argparse.Action, ), dict(
                            __call__=lambda a, p, n, v, o: getattr(n, a.dest).update(dict([v.split('=')])))),
                        default={
                            'dropout': 0.2,
                            'top_dims': None,
                            'bot_dims': None,
                        })

    # Optimizer
    parser.add_argument('--optim',
                        type=str,
                        default="Adam",
                        help="Specify optimizer")
    parser.add_argument('--lr',
                        type=float,
                        default=0.025,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help="Weight decay factor")
    # Sampler
    parser.add_argument('--sampler',
                        type=str,
                        choices=['weighted', 'simple'],
                        default='weighted',
                        help="Select the sample strategy")

    # Input Format & Task
    parser.add_argument('--task',
                        type=str,
                        choices=['retrieval', 'rank', 'sess'],
                        required=True,
                        help="specify the task type for using different trainer")

    parser.add_argument('--format',
                        type=str,
                        default='["u","i","r"]',)

    parser.add_argument('--n_targets',
                        type=int,
                        default=1)
    parser.add_argument('--sep',
                        default=",",
                        type=str)

    parser.add_argument('--header',
                        default=False,
                        type=str2bool)

    # Training
    parser.add_argument('--train',
                        type=str,
                        help="Training file path")

    parser.add_argument('--val',
                        type=str,
                        help="Validation file path")

    parser.add_argument('--test',
                        type=str,
                        help="Test file path")
    parser.add_argument('--device',
                        type=str,
                        help="Specify device")
    parser.add_argument('--max_epochs',
                        type=int,
                        default=100,
                        help="Number of epoch during training")
    parser.add_argument('--es_patience',
                        type=int,
                        default=sys.maxsize,
                        help="Number of epoch trigger early stop")
    parser.add_argument('--es_by',
                        type=str,
                        choices=['loss', 'metric+', 'metric-'],
                        default='loss',
                        help="Early stop by loss or metric(highest or lowest)")
    parser.add_argument('--log_interval',
                        type=int,
                        default=5,
                        help="Log interval")
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="Batch size in one iteration")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="the core of dataloader use to fetch data.")
    parser.add_argument('--num_neg',
                        type=int,
                        default=5,
                        help="the num negative number sample")
    parser.add_argument('--context_length',
                        type=int,
                        default=0,
                        help="the size of sequence length used for batch")
    parser.add_argument('--pooling_opt',
                        type=str,
                        choices=['mean', 'max', 'none', 'concat'],
                        default='none',
                        help="specify the pooling strategy of sequence feature")

    parser.add_argument('--saved_option',
                        type=str,
                        choices=['embedding', 'model'],
                        default='model',
                        help="output is embedding or model checkpoint")

    parser.add_argument('--saved_path',
                        type=str,
                        default="./",
                        help="specify path to save model/embedding")

    parser.add_argument('--metrics',
                        type=str,
                        nargs="+",
                        choices=metric.__all__,
                        help="specify the metric which used to validate")

    # Preprocess
    parser.add_argument('--split_ratio',
                        type=float,
                        default=0.8,
                        help="Proportion for training and testing split")
    parser.add_argument('--time_order',
                        action='store_true',
                        help="whether to use time order to split dataset")
    parser.add_argument('--seed',
                        type=int,
                        default=2023,
                        nargs="+",
                        help="Seed (For reproducability)")
    # Verbosity
    parser.add_argument('--silence',
                        action='store_true',
                        default=False,
                        help="show more information")
    parser.add_argument('--hyper_opt',
                        action='store_true',
                        default=False,
                        help="whether to use hyperopt to search hyperparameter")

    parser.add_argument('--sp_enc_policy',
                        choices=['default', 'handle_unk', 'none'],
                        default='default',
                        help="the policy to encode the sparse feature")

    parser.add_argument('--embed_init',
                        type=str,
                        choices=['smore', 'rand', 'normal', 'uniform',
                                 'xavier_normal', 'xavier_uniform',
                                 'kaiming_normal', 'kaiming_uniform'],
                        default='smore',
                        help="the method to initialize the embedding")

    # Pretrain
    parser.add_argument('--n_ctx',
                        type=int,
                        help="num of token in context")
    parser.add_argument('--CLIP_path',
                        type=str,
                        help="specify path to CLIP model")
    parser.add_argument('--text_json',
                        type=str,
                        default="./",
                        help="specify path to CLIP model")
    parser.add_argument('--text_by_idx',
                        type=int,
                        default=0)

    parser.add_argument('--pretrain',
                        type=str,
                        nargs="+",
                        default=[],
                        help="Pretrain file path")
    # parser.add_argument('--p_format',
    # type=str,
    # help="Pretrain file path")
    args = parser.parse_args()

    # ---------------

    args.sep = args.sep if args.sep != '\\t' else "\t"

    if args.embed_dim <= 0:  # embed_dim=None -> auto
        args.embed_dim = 'auto'

    args.mlp_params = {
        k: ([int(s) for s in dash_separated_strs(v)]
            if isinstance(v, str) else v)
        for k, v in args.mlp_params.items()
    }

    args.saved_path = os.path.join(args.saved_path, f"{args.model}.ckpt") \
        if args.saved_option == 'model' \
        else os.path.join(args.saved_path, f"{args.model}.emb")

    return args
