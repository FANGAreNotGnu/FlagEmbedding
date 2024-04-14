CONFIG=/media/code/FlagEmbedding/FlagEmbedding/baai_general_embedding/pipeline/configs/fever_config.yaml
PREPARE=false
DEDUP=false
MINING=false
FINETUNE=true
EVALUATE=true
GATHER=true
#STEPS=( 1000 2000 4000 6000 8000 16000 )
STEPS=( 32000 )

if $PREPARE; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.prepare_ir \
        --cfg $CONFIG
fi


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
