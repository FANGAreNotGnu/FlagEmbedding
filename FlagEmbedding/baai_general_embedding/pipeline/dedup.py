import argparse
import numpy as np
import random
from tqdm import tqdm

from FlagEmbedding import FlagModel
from FlagEmbedding.dedup.load_data import load_positive_pairs, save_positive_pairs
from FlagEmbedding.baai_general_embedding.pipeline.dedup import deduplicate_pairs
from FlagEmbedding.baai_general_embedding.pipeline.utils import load_positive_pairs, save_positive_pairs, get_flag_format_dataset_path, get_deduplicated_dataset_path


# pairwise
def remove_easy_positive(pairs, model, kept_pct):
    """
    pairs: [{"query":<query>,"doc":<doc>}, ...]
    kept_pct: kept percentage
    """
    assert kept_pct > 0 and kept_pct <= 1

    scores = []
    querys = [pair["query"] for pair in pairs]
    docs = [pair["doc"] for pair in pairs]
    embeddings_1 = model.encode(querys)
    embeddings_2 = model.encode(docs)
    scores = np.einsum('ij,ij->i', embeddings_1, embeddings_2)

    #for pair in tqdm(pairs):
    #    embeddings_1 = model.encode(pair["query"])
    #    embeddings_2 = model.encode(pair["doc"])
    #    score = embeddings_1 @ embeddings_2.T
    #    scores.append(score.item())
    top_indices = np.argsort(scores)[::-1]  # keep the hard positive (pairs with larger distance)
    kept_number = int(len(scores)*kept_pct)

    kept_pairs = [pairs[i] for i in top_indices[:kept_number]]

    return kept_pairs


# pairwise
def remove_random(pairs, kept_pct):
    """
    pairs: [{"query":<query>,"doc":<doc>}, ...]
    kept_pct: kept percentage
    """
    assert kept_pct > 0 and kept_pct <= 1

    kept_number = int(len(pairs)*kept_pct)
    kept_pairs = random.sample(pairs, kept_number)

    return kept_pairs


def deduplicate_pairs(input_file, output_file, kept_pct, model, dedup_mode="ep"):
    """
    dedup_mode:
        - ep: remove easy postive pairs
        - ran: remove random pairs
    """
    pairs = load_positive_pairs(input_file)

    if dedup_mode == "ep":
        kept_pairs = remove_easy_positive(pairs=pairs, model=model, kept_pct=kept_pct)
    elif dedup_mode == "ran":
        kept_pairs = remove_random(pairs=pairs, kept_pct=kept_pct)
    else:
        raise ValueError(f"dedup_mode is not supported: {dedup_mode}")

    save_positive_pairs(kept_pairs, output_file)
    print(f"dedup {len(kept_pairs)} out of {len(pairs)} pairs from {input_file} to {output_file}")


def dedup(config):
    dedup_model = config.dedup.model
    if dedup_model is None:
        dedup_model = config.pretrain.model

    model = FlagModel(dedup_model, query_instruction_for_retrieval=config.pretrain.query_instruction_for_retrieval)

    deduplicate_pairs(
        input_file=get_flag_format_dataset_path(config),
        output_file=get_deduplicated_dataset_path(config),
        model=model,
        kept_pct=config.dedup.kept_pct,
        dedup_mode=config.dedup.mode,
    )
