TRAIN_DATASET=lotte/pooled/dev/forum
HN_NAME=lotte_pooled_dev_forum_minedHN_easier
CORPUS_NAME=lotte_pooled_dev_forum_corpus
DEV_DATASET=lotte/pooled/dev/forum
TEST_DATASET=lotte/pooled/test/forum
OUTPUT_NAME=lotte_pooled_dev_forum_HN_easier_dense

range_for_sampling=500-1000
negative_number=7

MODEL_NAME=BAAI/bge-m3
LR=1e-6
EPOCHS=14
TEMP=0.02

OUTPUT_DIR=/media/code/FlagEmbedding/checkpoints/${OUTPUT_NAME}_${MODEL_NAME}_lr${LR}_${EPOCHS}e_t${TEMP}

PER_GPU_BS=8
MAX_Q=128
MAX_DOC=512

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine_irdatasets \
--model_name_or_path $MODEL_NAME \
--dataset_name $TRAIN_DATASET \
--range_for_sampling $range_for_sampling \
--negative_number $negative_number \
--candidate_pool /media/data/flagfmt/${CORPUS_NAME}.jsonl

