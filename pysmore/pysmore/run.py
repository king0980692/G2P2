import sys
import subprocess
import os
import argparse


def run_all():

    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="mf")
    args.add_argument('--sep', default=",",type=str)
    args.add_argument("--loss_fn", type=str, default="bpr")
    args.add_argument("--sampler", type=str, default="weighted")
    args.add_argument("--gpu", default=False,action="store_true")
    args.add_argument("--embed_dim", type=int, default=64)
    args.add_argument("--worker", type=int, default=0)
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--num_neg", type=int, default=4)
    args.add_argument("--update_times", type=int, default=200)
    args.add_argument("--train", type=str, required=True)
    args.add_argument("--test", type=str, required=True)
    args.add_argument("--embed", type=str, required=True)
    args.add_argument("--loader_type", type=str, required=True)
    args.add_argument('--format', type=str, default='[["u"],["i"],["y"]]')
    args = args.parse_args()
    args_dict = vars(args)

    train_valued_args = ['train','model', 'loss_fn', 'sampler', 'embed_dim', 'worker', 'batch_size', 'num_neg', 'update_times', 'embed', 'loader_type', 'format', 'sep' ]
    train_flag_args = ['gpu']

    train_args = [f'--{x}={args_dict[x]}' for x in args_dict if x in train_valued_args]
    train_args += [f'--{x}' for x in args_dict if (x in train_flag_args and args_dict[x]==True)]


    print(train_args)
    pysmore_train_cmd = ["pysmore_train"] + train_args

    ret = subprocess.run(pysmore_train_cmd)

    if ret.returncode != 0:
        sys.exit( 'Error: pysmore_train failed.')

    ## -----------

    rec_valued_args = ['train', 'test', 'embed','embed_dim', 'worker']
    rec_flag_args = []

    rec_args = [f'--{x}={args_dict[x]}' for x in args_dict if x in rec_valued_args]
    rec_args += [f'--{x}' for x in args_dict if (x in rec_flag_args and x==True)]

    print(rec_args)
    pysmore_rec_cmd = ["pysmore_rec"] + rec_args

    ret = subprocess.run(pysmore_rec_cmd)


    if ret.returncode != 0:
        sys.exit( 'Error: pysmore_rec failed.')

    ## -----------

    os.makedirs("./result", exist_ok=True)
    res_file_name = f"result/{args_dict['model']}.result"

    o_file = open(res_file_name, "w") 
    o_file.close()

    o_file = open(res_file_name, "a+") 
    for t in ['ui', 'ii']:
        for n in [5, 10, 20, 'all']:
            pysmore_eval_cmd = ['pysmore_eval'] + ["-R", f"{args_dict['embed']}.{t}.rec","-N", str(n)] 
            ret = subprocess.run(pysmore_eval_cmd, stdout=o_file)

            if ret.returncode != 0:
                sys.exit( 'Error: pysmore_eval failed.')

    ## --

    ret = subprocess.run(["cat", res_file_name])
    




if __name__ == '__main__':
    run_all()


