## TL;DR

Please refer [examples](examples/) for more information.

### 🔊 Recruiting
please refer [examples](examples/)  to help doing more experiment results for us to fill-up the whole table which we can equally compare with other recommendation algorithm in a fair standard.




## Prerequisite

### Install pysmore
Our repo is private for now, maybe we need to modify the command below in the future for public accessing.
```bash
pip install git+ssh://git@github.com/cnclabs/pysmore.git
```

### Compiler the cython (no need this step in the near future)
```bash
python3 setup.py build_ext --inplace
```



## Basic Usage

We provide three CLI entrypoints for finish recommendation.

### pysmore_train

```bash
usage: pysmore_train [-h] [--data DATA] [--saved_emb SAVED_EMB] [--seed SEED]
                     [--dim DIM] [--lr LR] [--weight_decay WEIGHT_DECAY]
                     [--sampler {weighted,simple}] [--gpu GPU]
                     [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]
                     [--eval_every EVAL_EVERY] [--update_times UPDATE_TIMES]
                     [--fetch_worker FETCH_WORKER] [--num_neg NUM_NEG]
                     [--dataset {ui,list}] [--data_dir DATA_DIR]
                     [--test_size TEST_SIZE] [--time_order]

```
For example:

```
pysmore_train \
    --dataset ui \
    --sampler 'weighted' \
    --dim $DIM \
    --fetch_worker $WORKER \
    --batch_size $BATCH \
    --data_dir $tr_data \
    --saved_emb $emb \
    --num_neg $NEG \
    --gpu $GPU \
    --update_times $UPDATE 
```

* `--dataset`: specify the input format, like user-item interactions or user-item history format 
* `--sampler`: specify the sampler method to update loss, for now we provide:
  * `simple`: sample with original user-item interaction 
  * `weighted`: sample with node degree 
* `--dim`: specify the embedding vector dimension
* `--fetch_worker`: specify the worker for dataloader to fetch data.
* `--batch_size`: specify the batch is composed with how many sample 
* `--data_dir`: specify the path of processed dataset
* `--save_emb`: specify the path of output embedding
* `--num_neg`: specify the number of negative sample
* `--gpu`: specify use gpu or not
* `--update_times`: specify update time for learning representation
 

### pysmore_rec

```
usage: pysmore_rec [-h] [--train TRAIN] [--test TEST] [--embed EMBED] [--emb_dim EMB_DIM]
                   [--cold_user COLD_USER] [--cold_item COLD_ITEM] [--num_test NUM_TEST]
                   [--worker WORKER] [--sim {dot,cosine}]

Argument Parser

options:
  -h, --help            show this help message and exit
  --train TRAIN         data.ui.train
  --test TEST           data.ui.test
  --embed EMBED         embeddding file
  --emb_dim EMB_DIM     emedding demensions
  --cold_user COLD_USER
                        to test cold user
  --cold_item COLD_ITEM
                        to test cold item
  --num_test NUM_TEST   # of sampled tests
  --worker WORKER       # of threads
  --sim {dot,cosine}    sim metric

```

### pysmore_eval

```
usage: pysmore_eval [-h] [-R RESULT] [-N TOP_K]

Argument Parser

optional arguments:
  -h, --help            show this help message and exit
  -R RESULT, --result RESULT
                        specify the hit result path
  -N TOP_K, --top-k TOP_K
                        the top-N of metric
```



