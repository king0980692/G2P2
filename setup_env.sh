pip install torch==2.1
pip install numpy==1.26.4

pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

pip install tqdm pandas ftfy regex gensim
pip install faiss-cpu loguru optuna tensorboard optuna tensorboard
pip intall torchtext==0.16.2
pip install "huggingface_hub[cli]"


cd pysmore
python3 setup.py develop
