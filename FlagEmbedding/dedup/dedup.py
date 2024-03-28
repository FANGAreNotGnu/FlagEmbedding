import argparse
import numpy as np
from tqdm import tqdm

from FlagEmbedding import FlagModel
from FlagEmbedding.dedup.load_data import load_positive_pairs, save_positive_pairs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default=None, type=str)
    parser.add_argument('--input_name', default=None, type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--kept_pct', default=None, type=float)

    return parser.parse_args()


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


def deduplicate_pairs(input_file, output_file, model, kept_pct):
    pairs = load_positive_pairs(input_file)
    kept_pairs = remove_easy_positive(pairs, model, kept_pct)
    save_positive_pairs(kept_pairs, output_file)
    print(f"dedup {len(kept_pairs)} out of {len(pairs)} pairs from {input_file} to {output_file}")


if __name__ == "__main__":
    args = get_args()

    model = FlagModel(args.model_name_or_path, query_instruction_for_retrieval="")

    input_file = f"/media/data/flagfmt/{args.input_name}.jsonl"
    output_file = f"/media/data/flagfmt/{args.output_name}.jsonl"

    deduplicate_pairs(
        input_file=input_file,
        output_file=output_file,
        model=model,
        kept_pct=args.kept_pct,
    )
