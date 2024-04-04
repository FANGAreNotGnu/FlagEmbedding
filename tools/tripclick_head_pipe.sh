CONFIG=/media/code/FlagEmbedding/FlagEmbedding/baai_general_embedding/pipeline/configs/tripclick_head_config.yaml
DEDUP=false
MINING=false
FINETUNE=true
EVALUATE=true
GATHER=true

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

if $GATHER; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.gather_results \
        --cfg $CONFIG
fi
