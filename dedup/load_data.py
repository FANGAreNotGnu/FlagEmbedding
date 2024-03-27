import os
import pandas as pd


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


if __name__ == "__main__":
    load_positive_pairs("/media/data/flagfmt/beir_nfcorpus_train.jsonl")
