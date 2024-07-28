import sys
from math import log
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import argparse

def pysmore_eval():

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('-R','--result', type=str, help='specify the hit result path')
    parser.add_argument('-N','--top-k', type=str, help='the top-N of metric')
    parser.add_argument('--print_opt',
                        default='csv',
                        choices=['tabular', 'csv'],
                        type=str, 
                        help='specify the output format')

    args = parser.parse_args()

    # rec_file = sys.argv[1]
    rec_file = args.result
    if args.top_k == 'all':
        at_k = sys.maxsize
    else:
        at_k = int(args.top_k)

    eval_num = 0.
    eval_hit = 0.
    eval_map = 0.
    eval_recall = 0.
    eval_ndcg = 0.
    eval_mrr = 0.

    with open(rec_file, 'r') as f:
        lines = f.readlines()
        eval_num = len(lines)
        for line in lines:
            line = line.rstrip().split(' ')
            uid = line[0]
            len_ans = min(at_k, int(line[1]))
            if len_ans == 0: continue

            weight_res = list(map(float, line[2:]))[:at_k]
            binary_res = [min(v, 1) for v in weight_res]

            # rannking scoring
            dcg, idcg = 0., 0.
            match, mAP = 0., 0.
            for pos, value in enumerate(weight_res, 1):
                if pos <= len_ans:
                    idcg += 1./log(pos+1, 2)
                if value:
                    dcg += value/log(pos+1, 2)
                    match += 1
                    mAP += match/pos
                    eval_mrr += 1/pos
            if idcg == 0:
                idcg = 1.
            # eval_num += 1
            if sum(binary_res):
                eval_hit += 1
            eval_map += mAP/min(at_k, len_ans)
            eval_ndcg += dcg/idcg
            eval_recall += sum(binary_res)/min(at_k, len_ans)

    k = "all" if at_k == sys.maxsize else at_k
    title = "u2i" if 'ui' in rec_file else "i2i"

    if args.print_opt == 'tabular':
        print (f'|        {title}              |')
        print ('|    -----------------    |')
        print ('|    HIT@{}:\t{:.6f}  |'.format(k, eval_hit/eval_num))
        print ('|    MRR@{}:\t{:.6f}  |'.format(k, eval_mrr/eval_num))
        print ('|    MAP@{}:\t{:.6f}  |'.format(k, eval_map/eval_num))
        print ('|   NDCG@{}:\t{:.6f}  |'.format(k, eval_ndcg/eval_num))
        print ('| RECALL@{}:\t{:.6f}  |'.format(k, eval_recall/eval_num))
        print("===========================")

    elif args.print_opt == 'csv':
        print(f'{title},HIT@{k},MRR@{k},MAP@{k},NDCG@{k},RECALL@{k}')
        print(f',{eval_hit/eval_num:.6f},{eval_mrr/eval_num:.6f},{eval_map/eval_num:.6f},{eval_ndcg/eval_num:.6f},{eval_recall/eval_num:.6f}')



if __name__ == '__main__':
    pysmore_eval()
