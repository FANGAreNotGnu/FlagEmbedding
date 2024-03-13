TRAIN_DATASET=lotte/pooled/dev/forum
HN_NAME=lotte_pooled_dev_forum_minedHN
CORPUS_NAME=lotte_pooled_dev_forum_corpus
DEV_DATASET=lotte/pooled/dev/forum
TEST_DATASET=lotte/pooled/test/forum
OUTPUT_NAME=lotte_pooled_dev_forum_HN_dense

MODEL_NAME=BAAI/bge-m3
LR=1e-6
EPOCHS=3
TEMP=0.02

OUTPUT_DIR=/media/code/FlagEmbedding/checkpoints/${OUTPUT_NAME}_${MODEL_NAME}_lr${LR}_${EPOCHS}e_t${TEMP}

PER_GPU_BS=8
MAX_Q=128
MAX_DOC=512

python3 /media/code/FlagEmbedding/tools/prepare_irdatasets.py \
  --dataset_names $TRAIN_DATASET $DEV_DATASET $TEST_DATASET

