import argparse
import os
import numpy as np

from FlagEmbedding.llm_embedder.src.utils import load_json
from FlagEmbedding.baai_general_embedding.pipeline.utils import load_config, get_result_path, get_model_save_path, seed_everything

def gather_result(config):
    model_save_path = get_model_save_path(config)

    metric_name = config.evaluate.main_metric

    def get_ckpt_iter(ckpt):
        return int(ckpt[11:])

    ckpts = os.listdir(model_save_path)
    ckpts = list(filter(lambda x: x[:11]=="checkpoint-", ckpts))

    ckpt_iters = [get_ckpt_iter(ckpt) for ckpt in ckpts]
    sorted_idx = np.argsort(ckpt_iters)

    # pretrain result
    result_path = get_result_path(config.pretrain.model, config)
    metrics = load_json(result_path)
    print(f"Pretraining result: {metrics[metric_name]}")

    # finetune result
    result_path = get_result_path(get_model_save_path(config), config)
    metrics = load_json(result_path)
    print(f"Finetuned result: {metrics[metric_name]}")

    # allckpts result
    for i in sorted_idx:
        print(ckpt_iters[i])

    for i in sorted_idx:
        ckpt = ckpts[i]
        ckpt_path = os.path.join(model_save_path, ckpt)
        result_path = get_result_path(ckpt_path, None)
        metrics = load_json(result_path)
        print(metrics[metric_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()
    config = load_config(args.cfg)

    seed_everything(config)

    gather_result(config)
