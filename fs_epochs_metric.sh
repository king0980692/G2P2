#!/bin/bash
set -ex

data=$1
type="support"
pysmore_train="python3 ./pysmore/pysmore/train.py"
pysmore_emb_pred="python3 ./pysmore/pysmore/emb_pred.py"
pysmore_eval="python3 ./pysmore/pysmore/eval.py"

declare -A data_abbr
data_abbr["Musical_Instruments"]="MI"
data_abbr["All_Beauty"]="BE"
data_abbr["reviews_Beauty_5"]="BE5"
data_abbr["Industrial_and_Scientific"]="IS"
data_abbr["Sports_and_Outdoors"]="SO"
data_abbr["Toys_and_Games"]="TG"
data_abbr["Arts_Crafts_and_Sewing"]="AC"

rm -f runs/${data}_${type}_mf_prompt.log
for model in res/${data}/*; do

  python3 zs_rec_amz.py --type $type --model $model --data_name $data
  # >/dev/null 2>&1

  $pysmore_emb_pred \
    --train tmp/${data_abbr[$data]}.train.u \
    --test tmp/${data_abbr[$data]}.test.u \
    --embed ./tmp/${data_abbr[$data]}_g2p2_t_$type.emb \
    --embed_dim 128
  # >/dev/null 2>&1

  epoch=$(echo "${model##*_}" | cut -d'.' -f1)
  res=$($pysmore_eval -R tmp/${data_abbr[$data]}_g2p2_t_$type.emb.ui.rec -N 10)

  HR=$(echo "${res}" | sed 1d | cut -d',' -f2)
  NDCG=$(echo "${res}" | sed 1d | cut -d',' -f5)
  # echo "0s,${epoch},${HR}, ${NDCG}"
  echo "type,epoch,n_ctx,lr,ckpt,HR,NDCG"
  echo "0s,"-","-","-",${epoch},${HR}, ${NDCG}" 

  ## -----------------

  n_ctx=4
  for lr in 0.0025; do
    for e in $(seq 5 10 50); do
      $pysmore_train \
        --model mf_prompt \
        --loss_fn BPR \
        --worker 0 \
        --batch_size 4 \
        --train tmp/${data_abbr[$data]}.$type.u \
        --device cpu \
        --saved_path ./tmp/ \
        --saved_option embedding \
        --max_epochs $e \
        --embed_dim 128 \
        --optim Adam \
        --lr $lr \
        --header 0 \
        --num_neg 1 \
        --sep '\t' \
        --format '["u","i","y","t"]' \
        --n_ctx $n_ctx \
        --text_json tmp/${data}_${type}_text.json \
        --pretrain tmp/${data_abbr[$data]}_g2p2_t_user_$type.emb tmp/${data_abbr[$data]}_g2p2_t_item_$type.emb \
        --CLIP_path $model \
        --task retrieval
      # >/dev/null 2>&1

      $pysmore_emb_pred \
        --train tmp/${data_abbr[$data]}.train.u \
        --test tmp/${data_abbr[$data]}.test.u \
        --embed ./tmp/mf_prompt.emb \
        --embed_dim 128 >/dev/null 2>&1

      epoch=$(echo "${model##*_}" | cut -d'.' -f1)
      res=$($pysmore_eval -R tmp/mf_prompt.emb.ui.rec -N 10)
      HR=$(echo "${res}" | sed 1d | cut -d',' -f2)
      NDCG=$(echo "${res}" | sed 1d | cut -d',' -f5)
      echo "type,epoch,n_ctx,lr,ckpt,HR,NDCG"
      echo "fs,$e,$n_ctx,$lr,${epoch},${HR}, ${NDCG}"

      exit 1
    done
  done
  echo ""
done
