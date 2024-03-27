import numpy as np


def remove_easy_positive(pairs, model, kept_pct):
    """
    pairs: [{"query":<query>,"doc":<doc>}, ...]
    kept_pct: kept percentage
    """
    assert kept_pct > 0 and kept_pct <= 1

    scores = []
    for pair in pairs:
        embeddings_1 = model.encode(pair["query"], normalize_embeddings=True)
        embeddings_2 = model.encode(pair["doc"], normalize_embeddings=True)
        score = embeddings_1 @ embeddings_2.T
        scores.append(score.item())
    top_indices = np.argsort(scores)[::-1]  # keep the hard positive (pairs with larger distance)
    kept_number = int(len(scores)*kept_pct)

    kept_pairs = [pairs[i] for i in top_indices[:kept_number]]

    return kept_pairs


