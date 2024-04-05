import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from FlagEmbedding.baai_general_embedding.finetune.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from FlagEmbedding.baai_general_embedding.finetune.data import TrainDatasetForEmbedding, EmbedCollator
from FlagEmbedding.baai_general_embedding.finetune.modeling import BiEncoderModel
from FlagEmbedding.baai_general_embedding.finetune.trainer import BiTrainer
from FlagEmbedding.baai_general_embedding.pipeline.utils import get_model_save_path, load_config, get_mined_dataset, seed_everything, save_config

logger = logging.getLogger(__name__)


def finetune():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    config = load_config(model_args.cfg)
    if model_args.steps is not None:
        config.optimization.max_steps = model_args.steps

    seed_everything(config)

    output_dir = get_model_save_path(config)

    # hardcode changes TODO: change in a for loop
    training_args.output_dir = output_dir
    training_args.learning_rate = config.optimization.learning_rate
    training_args.num_train_epochs = config.optimization.num_train_epochs
    training_args.max_steps = config.optimization.max_steps
    training_args.per_device_train_batch_size = config.optimization.per_device_train_batch_size
    training_args.max_grad_norm = config.optimization.max_grad_norm
    training_args.save_steps = config.optimization.save_steps
    training_args.fp16 = config.optimization.fp16
    training_args.warmup_ratio = config.optimization.warmup_ratio
    training_args.weight_decay = config.optimization.weight_decay

    training_args.lr_scheduler_type = config.optimization.lr_scheduler_type
    training_args.dataloader_drop_last = config.optimization.dataloader_drop_last
    training_args.disable_tqdm = config.optimization.disable_tqdm
    training_args.full_determinism = config.optimization.full_determinism
    training_args.logging_steps = config.optimization.logging_steps
    training_args.learning_rate = config.optimization.learning_rate

    training_args.temperature = config.optimization.temperature
    training_args.negatives_cross_device = config.optimization.negatives_cross_device
    training_args.fix_position_embedding = config.optimization.fix_position_embedding
    training_args.sentence_pooling_method = config.optimization.sentence_pooling_method
    training_args.normlized = config.optimization.normlized
    training_args.use_inbatch_neg = config.optimization.use_inbatch_neg
    # data_args
    data_args.train_data = get_mined_dataset(config=config, return_dir=True)
    data_args.train_group_size = config.data.train_group_size
    data_args.query_max_len = config.data.query_max_len
    data_args.passage_max_len = config.data.passage_max_len
    data_args.max_example_num_per_dataset = config.data.max_example_num_per_dataset
    #data_args.query_instruction_for_retrieval = config.data.query_instruction_for_retrieval
    data_args.query_instruction_for_retrieval = ""  # instruction is only used in evaluation
    data_args.passage_instruction_for_retrieval = config.data.passage_instruction_for_retrieval
    # model_args
    model_args.model_name_or_path = config.pretrain.model


    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Model Config: %s', model_config)

    model = BiEncoderModel(model_name=model_args.model_name_or_path,
                           normlized=training_args.normlized,
                           sentence_pooling_method=training_args.sentence_pooling_method,
                           negatives_cross_device=training_args.negatives_cross_device,
                           temperature=training_args.temperature,
                           use_inbatch_neg=training_args.use_inbatch_neg,
                           )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    # save config TODO: save only when global_rank==0
    save_config(config)


if __name__ == "__main__":
    finetune()
