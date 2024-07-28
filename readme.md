# G2P2 Rec

## Downstream adaption for recommendation


1. Prepare processed dataset

we uploaded the preocessed dataset and checkpoint on huggingface, use command below to download.

```bash
# preprocessed dataset
huggingface-cli download Leon-Chang/exp --repo-type dataset --local-dir ./tmp/

# checkpoint
mkdir -p res/Musical_Instruments/

huggingface-cli download Leon-Chang/g2p2_ckpts --repo-type dataset --local-dir ./res/Musical_Instruments/
```


1. Run the experiment
We provide the script to run the experiment, for example you can use below command to run the experiment on Musical_Instruments dataset
```bash
bash fs_epochs_metric.sh Musical_Instruments
```

## Pre-Train

Make sure the Amazon dataset is in `data` folder

for example Musical_Instruments dataset

```bash
mkdir data; cd data
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments.json.gz
```

and also its meta-data
```bash
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Musical_Instruments.json.gz
```

1. Preprocess the dataset

```bash
    python g2p2_ext/preprocess_amazon.py
```


2. Pre-Train the model

Note: this step might take 1 epoch per day, depend on your device.

if you want to reproduce the model, then just run it or you can use our model checkpoint, see more detail on below.

```bash
    python main_train_amazon.py
```

