from omegaconf import OmegaConf
import json
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


DATA_ROOT = "/media/data/flagfmt"
CKPT_ROOT = "/media/ragckpts"


def load_config(config_path):
    config = OmegaConf.load(config_path)

    return config


def load_positive_pairs(path):
    """
    path: a jsonl file with {"query": query_text, "pos": pos_texts} in each line.
    """
    data = pd.read_json(path_or_buf=path, lines=True)
    #print(data[:10])
    pairs = []
    for _, row in data.iterrows():
        for doc in row["pos"]:
            pairs.append({"query": row["query"], "doc": doc})
    print(f"Loaded {len(pairs)} positive pairs from {path}.")
    #print(pairs[:10])

    return pairs


def save_positive_pairs(pairs, path):
    """
    pairs: [{"query":<query>,"doc":<doc>}, ...]
    """
    saved_data = defaultdict(list)
    for pair in pairs:
        saved_data[pair["query"]].append(pair["doc"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w+') as f:
        for query, pos in tqdm(saved_data.items()):
            f.write(json.dumps({"query": query, "pos": pos}) + "\n")


def get_flag_format_dataset(config, return_path=True):
    if not return_path:
        return config.data.train.replace('/', '_'), config.data.val.replace('/', '_'), config.data.test.replace('/', '_')
    else:
        return f"{DATA_ROOT}/{config.data.train.replace('/', '_')}.jsonl", \
            f"{DATA_ROOT}/{config.data.val.replace('/', '_')}.jsonl", \
            f"{DATA_ROOT}/{config.data.test.replace('/', '_')}.jsonl"


def get_deduplicated_dataset(config, return_path=True):
    dedup_name = f"{config.data.train.replace('/', '_')}/dedup_"

    model_name = config.dedup.model if config.dedup.model is not None else config.pretrain.model
    dedup_name += model_name.replace('/', '_')
    dedup_name += f"_kept{config.dedup.kept_pct}_{config.dedup.mode}"

    if not return_path:
        return dedup_name
    else:
        return f"{DATA_ROOT}/{dedup_name}.jsonl"


def get_corpus_name(config, return_path=True, return_split="train"):
    if return_split == "train":
        corpus_name = f"{config.data.train.replace('/', '_')}_corpus"
    elif return_split == "val":
        corpus_name = f"{config.data.val.replace('/', '_')}_corpus"
    elif return_split == "test":
        corpus_name = f"{config.data.test.replace('/', '_')}_corpus"
    else:
        raise ValueError(f"Invalid return_split: {return_split}")

    if return_path:
        return f"{DATA_ROOT}/{corpus_name}.jsonl"
    else:
        return corpus_name


def get_mined_dataset(config, return_path=True, return_dir=False):
    mined_name = get_deduplicated_dataset(config, return_path=False)
    mined_name += f"-mined_{config.mining.mode}"
    if config.mining.mode == "hard":
        mined_name += f"_{config.mining.range_for_sampling}"
    mined_name += f"_{config.mining.negative_number}"
    if return_dir:
        return f"{DATA_ROOT}/{mined_name}"
    elif return_path:
        return f"{DATA_ROOT}/{mined_name}/mined_data.jsonl"
    else:
        return mined_name


def get_model_save_path(config, return_path=True):
    model_save_name = get_mined_dataset(config, return_path=False)
    model_save_name += f"-{config.optimization.learning_rate}_{config.optimization.num_train_epochs}e"
    model_save_name += f"-{config.optimization.temperature}T_{config.optimization.weight_decay}wd"
    if return_path:
        return f"{CKPT_ROOT}/{model_save_name}"
    else:
        return model_save_name


def get_model_ckpt(config, ckpt_step, return_path=True):
    model_save_path = get_model_save_path(config, return_path=True)
    if ckpt_step == "allckpts":
        ckpt_steps = os.listdir(model_save_path)
        ckpt_steps = filter(lambda x: x[:11]=="checkpoint-", ckpt_steps)
        ckpt_steps = [int(x[11:]) for x in ckpt_steps]
        return [get_model_ckpt(config, step) for step in ckpt_steps]
    else:
        return os.path.join(model_save_path, f"checkpoint-{ckpt_step}")


def get_result_path(encoder_name, config):
    if os.path.exists(encoder_name):
        return os.path.join(encoder_name, "results.json")
    else:
        return f"{DATA_ROOT}/{config.data.test}-{encoder_name.replace('/', '_')}-results.json"


def save_config(config, save_path=None):
    if save_path is None:
        save_path = os.papth.join(get_model_save_path(config), "rag_config.yaml")

    print(f"Saving RAG config file at {save_path}...")
    OmegaConf.save(config, save_path)
    

def seed_everything(config):
    import numpy as np
    import os
    import random
    import torch

    seed = config.seed

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
