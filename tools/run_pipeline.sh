TRAIN_DATASET=beir/fiqa/train
HN_NAME=beir_fiqa_train_minedHN
DEV_DATASET=beir/fiqa/dev
TEST_DATASET=beir/fiqa/test
OUTPUT_NAME=fiqa_HN_dense

MODEL_NAME=BAAI/bge-m3
LR=1e-5
EPOCHS=3
TEMP=0.02

OUTPUT_DIR=${OUTPUT_NAME}_${MODEL_NAME}_lr${LR}_${EPOCHS}e_t${TEMP}

PER_GPU_BS=8
MAX_Q=128
MAX_DOC=512

python3 /media/code/FlagEmbedding/tools/prepare_irdatasets.py \
  --dataset_names $TRAIN_DATASET $DEV_DATASET $TEST_DATASET

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine_irdatasets \
--model_name_or_path BAAI/bge-m3 \
--dataset_name $TRAIN_DATASET \
--range_for_sampling 10-100 \
--negative_number 3

torchrun --nproc_per_node 8 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir /media/code/FlagEmbedding/outputs/${OUTPUT_DIR} \
--model_name_or_path $MODEL_NAME \
--train_data /media/data/flagfmt/${HN_NAME} \
--learning_rate ${LR} \
--fp16 \
--num_train_epochs ${EPOCHS} \
--per_device_train_batch_size ${PER_GPU_BS} \
--dataloader_drop_last TRUE \
--normlized True \
--temperature $TEMP \
--query_max_len ${MAX_Q} \
--passage_max_len ${MAX_DOC} \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval ""

python -m FlagEmbedding.baai_general_embedding.finetune.eval_irdatasets \
--encoder ${OUTPUT_DIR} \
--fp16 \
--add_instruction \
--k 100 \
--max_passage_length $MAX_DOC \
--eval_data_name $TEST_DATASET
