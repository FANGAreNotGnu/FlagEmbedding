CONFIG=/media/code/FlagEmbedding/FlagEmbedding/baai_general_embedding/pipeline/configs/nfcorpus_config.yaml
DEDUP=true
MINING=true
FINETUNE=true
EVALUATE=true
GATHER=true
EPOCHS=( 15 25 50 100 150 200 )

if $DEDUP; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.dedup \
        --cfg $CONFIG
fi

if $MINING; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.mining \
        --cfg $CONFIG
fi

if $FINETUNE; then
    for epochs in ${EPOCHS[*]} 
    do
        torchrun --nproc_per_node 8 \
            -m  FlagEmbedding.baai_general_embedding.pipeline.finetune \
            --cfg $CONFIG \
            --epochs $epochs
    done
fi

if $EVALUATE; then
    for epochs in ${EPOCHS[*]} 
    do       
        python -m FlagEmbedding.baai_general_embedding.pipeline.evaluate \
            --cfg $CONFIG \
            --epochs $epochs
    done
fi

if $GATHER; then
    python -m FlagEmbedding.baai_general_embedding.pipeline.gather_epoch_results \
        --cfg $CONFIG \
        --epochs ${EPOCHS[*]} 
fi
