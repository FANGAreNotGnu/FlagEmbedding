import argparse
import numpy as np
import random
from tqdm import tqdm

from FlagEmbedding import FlagModel
from FlagEmbedding.baai_general_embedding.pipeline.mining import batch_search, create_index
from FlagEmbedding.baai_general_embedding.pipeline.utils import load_config, load_positive_pairs, save_positive_pairs, get_flag_format_dataset, get_deduplicated_dataset, seed_everything


# pairwise
def remove_easy_positive(pairs, model, kept_pct, remove_hard=False):
    """
    pairs: [{"query":<query>,"doc":<doc>}, ...]
    kept_pct: kept percentage
    """
    assert kept_pct > 0 and kept_pct <= 1

    scores = []
    querys = [pair["query"] for pair in pairs]
    docs = [pair["doc"] for pair in pairs]
    embeddings_1 = model.encode(querys)  # TODO: use encode_query?
    embeddings_2 = model.encode(docs)
    scores = np.einsum('ij,ij->i', embeddings_1, embeddings_2)

    #for pair in tqdm(pairs):
    #    embeddings_1 = model.encode(pair["query"])
    #    embeddings_2 = model.encode(pair["doc"])
    #    score = embeddings_1 @ embeddings_2.T
    #    scores.append(score.item())
    if not remove_hard:
        top_indices = np.argsort(scores)[::-1]  # by default, keep the hard positive (pairs with larger distance)
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


# TODO: improve efficiency with FAISS
def get_upper_sim(model, texts, is_query=False):
    """
    texts: list of string
    """
    if is_query:
        embeddings = model.encode(texts)
    else:
        embeddings = model.encode_query(texts)
    similarities = embeddings @ embeddings.T
    similarities = np.triu(similarities, k=1)  # keep upper mat without diag
    return similarities


def search_two_step(
        query_index,
        doc_index,
        query_embeddings,
        doc_embeddings,
        query_threshold,
        doc_threshold,
        topk=256,
        batch_size=256,
    ):
    N = len(query_embeddings)
    inxs_to_remove = []
    inxs_not_to_remove = []
    print(f"two stage deduplicates search:")
    for start_index in tqdm(range(0, N, batch_size), desc="Batches", disable=N < batch_size):
        batch_query = query_embeddings[start_index:start_index + batch_size]
        queries_scores, queries_inxs = query_index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        batch_doc = doc_embeddings[start_index:start_index + batch_size]
        docs_scores, docs_inxs = doc_index.search(np.asarray(batch_doc, dtype=np.float32), k=topk)

        for i in range(start_index, start_index + len(queries_scores)):
            query_scores = queries_scores[i-start_index]
            query_inxs = queries_inxs[i-start_index]
            query_inxs = query_inxs[query_scores > query_threshold]

            doc_scores = docs_scores[i-start_index]
            doc_inxs = docs_inxs[i-start_index]
            doc_inxs = doc_inxs[doc_scores > doc_threshold]

            dup_inxs = query_inxs[np.isin(query_inxs, doc_inxs)]

            dup_inxs = dup_inxs[dup_inxs > i]  # use upper triangular

            if dup_inxs.size > 0:
                inxs_to_remove += dup_inxs.tolist()
                inxs_not_to_remove += [i]


    '''
    for i in tqdm(range(N)):
        query_scores, query_inxs = query_index.search(np.asarray(query_embeddings[i:i+1, :], dtype=np.float32), k=topk)
        doc_scores, doc_inxs = doc_index.search(np.asarray(doc_embeddings[i:i+1, :], dtype=np.float32), k=topk)
        
        query_inxs = query_inxs[query_scores > query_threshold]
        doc_inxs = doc_inxs[doc_scores > doc_threshold]
        dup_inxs = query_inxs[np.isin(query_inxs, doc_inxs)]
        dup_inxs = dup_inxs[dup_inxs > i]  # use upper triangular

        if dup_inxs.size > 0:
            inxs_to_remove += dup_inxs.tolist()
            inxs_not_to_remove += [i]
    
    #inxs_to_remove = set(inxs_to_remove) - set(inxs_not_to_remove)
    '''
    return inxs_to_remove




def two_step_dedup(pairs, model, query_threshold, doc_threshold):
    querys = [pair["query"] for pair in pairs]
    #query_embeddings = model.encode_query(texts)
    query_embeddings = model.encode(querys)
    query_index = create_index(query_embeddings, use_gpu=False)

    docs = [pair["doc"] for pair in pairs]
    doc_embeddings = model.encode(docs)
    doc_index = create_index(doc_embeddings, use_gpu=False)

    index_to_remove = search_two_step(
        query_index=query_index, 
        doc_index=doc_index, 
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        query_threshold=query_threshold,
        doc_threshold=doc_threshold,
    )

    # TODO: change to use FAISS
    #q_sim = get_upper_sim(model, querys)  # TODO: use encode_query?
    #d_sim = get_upper_sim(model, docs)
    #dup_pairs = (q_sim > query_threshold) * (d_sim > doc_threshold)
    #print(dup_pairs)
    #dup_pairs = list(zip(*np.where(dup_pairs > 0)))
    #print(dup_pairs)

    #remove ones with smaller index
    #index_to_remove = [p[0] for p in dup_pairs]
    #index_not_to_remove = [p[1] for p in dup_pairs]
    #index_to_remove = list(set(index_to_remove) - set(index_not_to_remove))
    kept_pairs = [pair for idx, pair in enumerate(pairs) if idx not in index_to_remove]

    print(f"Kept {len(kept_pairs)} pairs from {len(pairs)} pairs in two_step_dedup")

    return kept_pairs


def deduplicate_pairs(input_file, output_file, kept_pct, model, query_threshold, doc_threshold, dedup_mode="ep"):
    """
    dedup_mode:
        - ep: remove easy postive pairs
        - ran: remove random pairs
    """
    pairs = load_positive_pairs(input_file)

    if dedup_mode == "ep":
        kept_pairs = remove_easy_positive(pairs=pairs, model=model, kept_pct=kept_pct)
    elif dedup_mode == "hp":
        kept_pairs = remove_easy_positive(pairs=pairs, model=model, kept_pct=kept_pct, remove_hard=True)
    elif dedup_mode == "ran":
        kept_pairs = remove_random(pairs=pairs, kept_pct=kept_pct)
    else:
        raise ValueError(f"dedup_mode is not supported: {dedup_mode}")
        
    kept_pairs = two_step_dedup(kept_pairs, model, query_threshold, doc_threshold)

    save_positive_pairs(kept_pairs, output_file)
    print(f"dedup {len(kept_pairs)} out of {len(pairs)} pairs from {input_file} to {output_file}")


def dedup(config):
    dedup_model = config.dedup.model
    if dedup_model is None:
        dedup_model = config.pretrain.model

    model = FlagModel(dedup_model, query_instruction_for_retrieval=config.data.query_instruction_for_retrieval)

    deduplicate_pairs(
        input_file=get_flag_format_dataset(config)[0],
        output_file=get_deduplicated_dataset(config),
        model=model,
        kept_pct=config.dedup.kept_pct,
        dedup_mode=config.dedup.mode,
        query_threshold=config.dedup.two_step.query_threshold,
        doc_threshold=config.dedup.two_step.doc_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()
    config = load_config(args.cfg)

    seed_everything(config)

    dedup(config)
