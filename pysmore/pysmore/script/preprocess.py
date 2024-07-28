import random
import os
from datetime import datetime
from tqdm import tqdm
import subprocess 
import argparse
import logging as logger
from collections import namedtuple
import json
import bisect


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value
def dash_separated_strs(value):
    try:
        vals = value.split("-")
    except ValueError:
        raise argparse.ArgumentTypeError(
            "%s is not a valid dash separated list of ints" % value
        )

    return value


def get_delimiter(file_path: str) -> str:
    import csv
    with open(file_path, 'r') as csvfile:
        delimiter = str(csv.Sniffer().sniff(csvfile.read()).delimiter)
        return delimiter

def _split(args, fields_list, in_fields):

    ## Time field
    time_idx_in_format = -1
    time_key_in_format = ""

    try:
        time_idx_in_format = fields_list.index('time')
        time_key_in_format = 'time'
    except ValueError:
        pass
    try:
        time_idx_in_format = fields_list.index('t')
        time_key_in_format = 't'
    except ValueError:
        pass

    assert time_idx_in_format != -1, "Not a valid time format" 



    ## Weight(Rating) field

    weight_idx_in_format = -1
    weight_key_in_format = ""

    try:
        weight_idx_in_format = fields_list.index('weight')
        weight_key_in_format = 'weight'
    except ValueError:
        pass
    try:
        weight_idx_in_format = fields_list.index('w')
        weight_key_in_format = 'w'
    except ValueError:
        pass

    try:
        weight_idx_in_format = fields_list.index('rating')
        weight_key_in_format = 'rating'
    except ValueError:
        pass
    try:
        weight_idx_in_format = fields_list.index('r')
        weight_key_in_format = 'r'
    except ValueError:
        pass
    assert weight_idx_in_format != -1, "Not a valid weight format" 

    # time_format_func = datetime.fromtimestamp
    time_format_func = lambda x:str(x) if args.time_format=="" else datetime.strptime(x,args.time_format)
    time_format_lambda_func = lambda x: int(x) if args.time_format=="" else x

    weight_format_func = int
    weight_format_lambda_func = lambda x: float(x)>=args.binarize_thld if args.binarize else x

    ## ---

    for _f in fields_list:
        globals()[f"{_f}_map"] = {} 
        globals()[f"{_f}_rvmap"] = {} 

    special_cut = -1
    if args.special_cut != "":
        special_cut = datetime.strptime(args.special_cut, args.time_format)

        print("Special Cut: ", special_cut)
    ## ---

    records = []
    with open(args.input) as f:
        # for line in tqdm(f, total=total_lines):
        for line in tqdm(f):

            line = line.rstrip().split(args.sep)
            line_dict = in_fields(*line)._asdict()

            if args.index:
                for id, _f in enumerate(line_dict):
                    if id in [weight_idx_in_format, time_idx_in_format]:
                        continue
                    if line_dict[_f] not in globals()[f"{_f}_map"]:
                        globals()[f"{_f}_map"][line_dict[_f]] = len(globals()[f"{_f}_map"])
                        globals()[f"{_f}_rvmap"][len(globals()[f"{_f}_map"])-1] = line_dict[_f]
                    line_dict[_f] = str(globals()[f"{_f}_map"][line_dict[_f]])


            line_dict[time_key_in_format] = time_format_func(
                time_format_lambda_func(line_dict[time_key_in_format])
            )

            line_dict[weight_key_in_format] = str(weight_format_func(
                weight_format_lambda_func(line_dict[weight_key_in_format])
            ))

            # if args.special_cut !="" and line_dict[time_key_in_format] < special_cut:
                # continue
            records.append([*line_dict.values()])

    if args.split_opt == "time":
        # records = sorted(records, key=lambda x: int(x[time_idx_in_format]))
        records = sorted(records, key=lambda x: x[time_idx_in_format])
    elif args.split_opt == "random":
        random.shuffle(records)
    elif args.split_opt == "none":
        pass

    if args.split_cut == "":
        _cut = int(len(records)*args.split_ratio)
        train_records = records[:_cut]
        test_records = records[_cut:]
    else:
        
        split_cut = datetime.strptime(args.split_cut, args.time_format)
        lower = bisect.bisect_right([r[time_idx_in_format] for r in records], special_cut)
        upper = bisect.bisect_left([r[time_idx_in_format] for r in records], split_cut)

        train_records = records[lower:upper]
        test_records = records[upper:]


    ## dump index
    if args.index:
        out_index = os.path.join(args.output, ".".join([args.name, "index.json"]))
        print(out_index)

        output = []
        for _f in fields_list:
            if _f in [weight_key_in_format, time_key_in_format]:
                continue
            output.append(f'{_f}, {str(len(globals()[f"{_f}_map"]))}')
            output.append(json.dumps(globals()[f"{_f}_map"], separators=(',', ':')))
            output.append(json.dumps(globals()[f"{_f}_rvmap"], separators=(',', ':')))
            # for _k in globals()[f"{_f}_map"]:
                # output.append("\t".join([_k, str(globals()[f"{_f}_map"][_k]) ]))

        with open(out_index, 'w') as f:
            f.write("\n".join(output))

    return train_records, test_records



def process_file(args):

    if args.sep == "auto":
        args.sep = get_delimiter(args.input)
    if args.sep == "\\t":
        args.sep = "\t"

    logger.info("Seperator: {}".format(args.sep))

    # total_lines = int(subprocess.check_output(f"wc -l {args.input}").split()[0])

    ## Input Format Tuple
    in_fields = namedtuple("in_fields", args.in_fields.split('-'))
    fields_list = args.in_fields.split('-')

                
    train_records, test_records = _split(args, fields_list, in_fields)

    logger.info("Output to folder : {}".format(args.output))

    ## Output Format Tuple
    out_fields = args.out_fields.split('-')

    filter_fields = ["r", "rate", "time", "t"]

    # Train
    out_train = os.path.join(args.output, ".".join([args.name, "train.tsv"]))
    print(out_train, len(train_records))
    with open(out_train, 'w') as f:
        for line in train_records:
            line_dict = in_fields(*line)._asdict()
            line = [f"{_f}{line_dict[_f]}" if _f not in filter_fields and not args.index else f"{line_dict[_f]}" for _f in out_fields ]

            f.write("\t".join(line)+"\n")
            
    # Test
    out_test = os.path.join(args.output, ".".join([args.name, "test.tsv"]))
    print(out_test, len(test_records))
    with open(out_test, 'w') as f:
        for line in test_records:
            line_dict = in_fields(*line)._asdict()
            line = [f"{_f}{line_dict[_f]}" if _f not in filter_fields and not args.index else f"{line_dict[_f]}" for _f in out_fields ]

            f.write("\t".join(line)+"\n")






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--name', type=str, default="pysmore")
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--in_fields', type=dash_separated_strs,required=True)
    parser.add_argument('--out_fields', type=dash_separated_strs)
    parser.add_argument('--time_format', default="", type=str,required=False)
    parser.add_argument('--sep', default="auto",type=str)
    parser.add_argument('--split_opt', choices=['time', 'random', 'none'], required=True)
    parser.add_argument('--split_ratio', type=float,default=0.8)
    parser.add_argument('--split_cut', type=str, default="")
    parser.add_argument('--special_cut', type=str, default="")
    parser.add_argument('--binarize', action='store_true')
    parser.add_argument('--binarize_thld', type=int, default=4)
    parser.add_argument('--index', action='store_true')
    args = parser.parse_args()

    ## process
    process_file(args)

