# Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting
We provide the implementation of G2P2 model, which is the source code for the SIGIR 2023 paper
"Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting". 

The repository is organised as follows:
- dataset/: the directory of data sets. Currently, it only has the dataset of Cora, if you want the *three processed Amazon datasets*, you can download and put them under this directory, the link is https://drive.google.com/drive/folders/1IzuYNIYDxr63GteBKeva-8KnAIhvqjMZ?usp=sharing. Besides, this link also contains the **4 pre-trained models**, under the directory of "pre-trained model".
- res/: the directory of saved models.
- bpe_simple_vocab_16e6.txt.gz: vocabulary for simple tokenization.
- data.py, data_graph.py: for data loading utilization.
- main_test.py, main_test_amazon.py: testing entrance for cora, testing entrance for Amazon datasets.
- main_train.py, main_train_amazon.py: pre-training entrance for cora, pre-training entrance for Amazon datasets.
- model.py, model_g_coop.py: model for pre-training, model for prompt tuning.
- multitask.py, multitask_amazon.py: task generator for cora, task generator for Amazon datasets.
- requirements.txt: the required packages.
- simple_tokenizer: a simple tokenizer.



# Leon's experiment

Only work for Amazon Music Instrument dataset.
Make sure the dataset is in `data` folder

1. Preprocess the dataset

```bash
    python g2p2_ext/preprocess_amazon.py
```


2. Pre-Train the model

```bash
    python main_train_amazon.py
```

3. For ZS task

```bash
    python zs_rec_amz.py
```

4. Use that embedding to predict the result
    
    
## Cite
	@inproceedings{wen2021augmenting,
		title = {Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting},
		author = {Wen, Zhihao and Fang, Yuan},
		booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
		year = {2023}
	}
