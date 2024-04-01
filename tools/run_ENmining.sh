TRAIN_DATASET=tripclick_train_head_dedup0.25
FLATTENED_NAME=tripclick_train_head

CORPUS_NAME=${FLATTENED_NAME}_corpus
NEGATIVE_NUMBER=500

python -m FlagEmbedding.baai_general_embedding.finetune.en_mine_irdatasets \
--dataset_name $TRAIN_DATASET \
--negative_number $NEGATIVE_NUMBER \
--candidate_pool /media/data/flagfmt/${CORPUS_NAME}.jsonl
