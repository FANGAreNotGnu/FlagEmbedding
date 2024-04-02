CONFIG=/media/code/FlagEmbedding/FlagEmbedding/baai_general_embedding/pipeline/default_config.yaml
DEDUP=false
MINING=false
FINETUNE=false
EVALUATE=true

if $DEDUP; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.dedup \
        --cfg $CONFIG
fi

if $MINING; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.mining \
        --cfg $CONFIG
fi

if $FINETUNE; then
    torchrun --nproc_per_node 8 \
    -m  FlagEmbedding.baai_general_embedding.pipeline.finetune \
        --cfg $CONFIG
fi

if $EVALUATE; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.evaluate \
        --cfg $CONFIG
fi
