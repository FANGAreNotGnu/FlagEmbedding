TRAIN_DATASET=beir/fiqa/train
DEV_DATASET=beir/fiqa/dev
TEST_DATASET=beir/fiqa/test
FLATTENED_NAME=beir_fiqa_train

HN_NAME=${FLATTENED_NAME}_minedHN
CORPUS_NAME=${FLATTENED_NAME}_corpus
OUTPUT_NAME=${FLATTENED_NAME}_HN_easier_dense

MODEL_NAME=BAAI/bge-large-en-v1.5
LR=1e-6
EPOCHS=3
TEMP=0.02

OUTPUT_DIR=/media/code/FlagEmbedding/checkpoints/${OUTPUT_NAME}_${MODEL_NAME}_lr${LR}_${EPOCHS}e_t${TEMP}

PER_GPU_BS=8
MAX_Q=128
MAX_DOC=512

python3 /media/code/FlagEmbedding/tools/prepare_irdatasets.py \
  --dataset_names $TRAIN_DATASET $DEV_DATASET $TEST_DATASET
