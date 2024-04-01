from omegaconf import OmegaConf
import json
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


DATA_ROOT = "/media/data/flagfmt"


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
    dedup_name = f"{config.data.train.replace('/', '_')}-dedup_"

    model_name = config.dedup.model if config.dedup.model is not None else config.pretrain.model
    dedup_name += model_name.replace('/', '_')
    dedup_name += f"_kept{config.dedup.kept_pct}_{config.dedup.mode}"

    if not return_path:
        return dedup_name
    else:
        return f"{DATA_ROOT}/{dedup_name}.jsonl"


def get_corpus(config, return_path=True):
    corpus_name = f"{config.data.train.replace('/', '_')}_corpus"
    if return_path:
        return f"{DATA_ROOT}/{corpus_name}.jsonl"
    else:
        return corpus_name
