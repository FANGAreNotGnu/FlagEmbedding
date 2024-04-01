TRAIN_DATASET=tripclick_train_head_deRAN0.25
FLATTENED_NAME=tripclick_train_head
POSTFIX=_adjustedHN

CORPUS_NAME=${FLATTENED_NAME}_corpus
OUTPUT_NAME=${FLATTENED_NAME}${POSTFIX}

range_for_sampling=100-1100
negative_number=500

MODEL_NAME=BAAI/bge-large-en-v1.5
LR=1e-6
EPOCHS=8
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
--candidate_pool /media/data/flagfmt/${CORPUS_NAME}.jsonl \
--postfix $POSTFIX

