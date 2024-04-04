import argparse
import os
import numpy as np

from FlagEmbedding.llm_embedder.src.utils import load_json
from FlagEmbedding.baai_general_embedding.pipeline.utils import load_config, get_result_path, get_model_save_path, seed_everything

def gather_result(config, steps):
    results = []
    for step in steps:
        step = int(step)
        config.optimization.max_steps = step
        metric_name = config.evaluate.main_metric

        # pretrain result
        #result_path = get_result_path(config.pretrain.model, config)
        #metrics = load_json(result_path)
        #print(f"Pretraining result: {metrics[metric_name]}")

        # finetune result
        result_path = get_result_path(get_model_save_path(config), config)
        metrics = load_json(result_path)
        results.append(metrics[metric_name])
        print(f"Finetuned result: {metrics[metric_name]}")
    
    for step in steps:
        print(step)

    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--steps', nargs='+', required=True)
    args = parser.parse_args()
    config = load_config(args.cfg)

    seed_everything(config)

    gather_result(config, args.steps)
