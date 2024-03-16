TRAIN_DATASET=beir/fiqa/train
DEV_DATASET=beir/fiqa/dev
TEST_DATASET=beir/fiqa/test
FLATTENED_NAME=beir_fiqa_train

HN_NAME=${FLATTENED_NAME}_minedHN
CORPUS_NAME=${FLATTENED_NAME}_corpus
OUTPUT_NAME=${FLATTENED_NAME}_HN_easier_dense

range_for_sampling=10-20
negative_number=3

MODEL_NAME=BAAI/bge-large-en-v1.5
LR=1e-6
EPOCHS=6
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

