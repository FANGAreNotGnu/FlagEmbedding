import json
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


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


if __name__ == "__main__":
    load_positive_pairs("/media/data/flagfmt/beir_nfcorpus_train.jsonl")
