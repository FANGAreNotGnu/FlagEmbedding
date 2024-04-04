CONFIG=/media/code/FlagEmbedding/FlagEmbedding/baai_general_embedding/pipeline/configs/nfcorpus_config.yaml
DEDUP=false
MINING=false
FINETUNE=true
EVALUATE=true
GATHER=true
STEPS=( 128 256 512 1024 2048 4096 8192 )
#STEPS=( 128 )

if $DEDUP; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.dedup \
        --cfg $CONFIG
fi

if $MINING; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.mining \
        --cfg $CONFIG
fi

if $FINETUNE; then
    for steps in ${STEPS[*]} 
    do
        torchrun --nproc_per_node 8 \
            -m  FlagEmbedding.baai_general_embedding.pipeline.finetune \
            --cfg $CONFIG \
            --steps $steps
    done
fi

if $EVALUATE; then
    for steps in ${STEPS[*]} 
    do       
        python -m FlagEmbedding.baai_general_embedding.pipeline.evaluate \
            --cfg $CONFIG \
            --steps $steps
    done
fi

if $GATHER; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.gather_step_results \
        --cfg $CONFIG \
        --steps ${STEPS[*]} 
fi
