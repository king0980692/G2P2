# Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting
We provide the implementaion of G2P2 model, which is the source code for the SIGIR 2023 paper
"Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting". 

The repository is organised as follows:
- dataset/: the directory of data sets. 
- res/: the directory of saved models.
- bpe_simple_vocab_16e6.txt.gz: vocabulary for simple tokenization.
- data.py, data_graph.py: for data loading utilization.
- main_test.py, main_test_amazon.py: testing entrance for cora, testing entrance for Amazon datasets.
- main_train.py, main_train_amazon.py: pre-training entrance for cora, pre-training entrance for Amazon datasets.
- model.py, model_g_coop.py: model for pre-training, model for prompt tuning.
- multitask.py, multitask_amazon.py: task generator for cora, task generator for Amazon datasets.
- requirements.txt: the required packages.
- simple_tokenizer: a simple tokenizer.


# For pre-train:
On Cora dataset,

    python main_train.py 

If on Amazon datasets, it should be:

    python main_train_amazon.py

# For testing:
On Cora dataset,

    python main_test.py 

If on Amazon datasets, it should be:

    python main_test_amazon.py