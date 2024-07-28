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
    # Parse argument
    parser = argparse.ArgumentParser()
    ## Model
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help="Select Model")

    parser.add_argument('--embed_dim',
                        type=int,
                        default=0,
                        help="Dimension for embedding")



    ## Input Format & Task
    parser.add_argument('--task',
                        type=str,
                        choices=['retrieval', 'rank', 'dlrm'],
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
                        type=bool)

    ## Training
    parser.add_argument('--train',
                        type=str,
                        help="Train file path")
    parser.add_argument('--test',
                        type=str,
                        help="Test file path")
    parser.add_argument('--device',
                        type=str,
                        help="Specify device")

    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="Batch size in one iteration")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="the core of dataloader use to fetch data.")
    parser.add_argument('--ckpt_path',
                        type=str,
                        default="",
                        help="path to load model checkpoint")
    parser.add_argument('--saved_path',
                        type=str,
                        default="",
                        help="path to save the result")

    parser.add_argument('--metrics',
                        type=str,
                        nargs="+",
                        choices=['AUC','Accuracy','u_F1','M_F1'],
                        help="specify the metric you want to use")


    ## Preprocess
    parser.add_argument('--test_size',
                        type=float,
                        default=0.1,
                        help="Proportion for training and testing split")
    parser.add_argument('--time_order',
                        action='store_true',
                        help="Proportion for training and testing split")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed (For reproducability)")

    ## Verbosity 
    args = parser.parse_args()

    ## ---------------

    args.sep = args.sep if args.sep != '\\t' else "\t"



    

    return args
