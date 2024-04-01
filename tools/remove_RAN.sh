FLATTENED_NAME=tripclick_train_head

MODEL_NAME=BAAI/bge-large-en-v1.5
KEPT_PCT=0.25
INPUT_NAME=${FLATTENED_NAME}
OUTPUT_NAME=${FLATTENED_NAME}_deRAN${KEPT_PCT}

python -m FlagEmbedding.dedup.dedup \
--model_name_or_path $MODEL_NAME \
--input_name $INPUT_NAME \
--output_name $OUTPUT_NAME \
--kept_pct $KEPT_PCT \
--dedup_mode ran
