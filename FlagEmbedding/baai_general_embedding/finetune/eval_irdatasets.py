import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import HfArgumentParser
from FlagEmbedding import FlagModel
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    
    max_query_length: int = field(
        default=128,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=1024,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=128,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )
    eval_data_root: str = field(
        default="/media/data/flagfmt/",
        metadata={'help': 'Root to data.'}
    )
    eval_data_name: str = field(
        default="lotte_science_test_forum",
        metadata={'help': 'Name of data.'}
    )


def index(model: FlagModel, corpus: datasets.Dataset, batch_size: int = 256, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        corpus_embeddings = model.encode_corpus(corpus["text"], batch_size=batch_size, max_length=max_length)
        dim = corpus_embeddings.shape[-1]
        
        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    #if model.device == torch.device("cuda"):
    #    # co = faiss.GpuClonerOptions()
    #    co = faiss.GpuMultipleClonerOptions()
    #    co.useFloat16 = True
    #    # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
    #    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(model: FlagModel, queries: datasets, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries(queries["query"], batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    
    
def evaluate(preds, labels, scores, cutoffs):
    """
    Evaluate MRR and Recall at cutoffs.
    preds: (N, args.k) of texts
    labels: (N, <dynamic>num_pos) of texts
    scores: (N, args.k) of rel scores in [0,1]
    """
    N = len(preds)
    max_cutoff = len(preds[0])
    assert max(cutoffs) <= max_cutoff, f"k {max(cutoffs)} and k {max_cutoff}"

    assert len(preds[-1]) == max_cutoff, f"shape {preds[-1]} and shape {max_cutoff}"
    assert len(labels) == N, f"shape {len(labels)} and shape {N}"
    assert len(scores) == N, f"shape {len(scores)} and shape {N}"
    assert len(scores[0]) == max_cutoff, f"shape {len(scores[0])} and shape {max_cutoff}"

    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)

    # Recall, Success, and NDCG
    recalls = np.zeros(len(cutoffs))
    successes = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
            successes[k] += int(len(recall) > 0)

    # NDCG
    ndcgs = np.zeros(len(cutoffs))
    true_labels = []
    for pred, label in zip(preds, labels):
        true_labels_per_query = []
        for pred_text in pred:
            if pred_text in label:
                true_labels_per_query.append(1)
            else:
                true_labels_per_query.append(0)
        true_labels.append(true_labels_per_query)
    for k, cutoff in enumerate(cutoffs):
            if cutoff <= 1:
                ndcgs[k] = -1
            else:
                ndcgs[k] = ndcg_score(true_labels, scores, k=cutoff)

    recalls /= len(preds)
    successes /= len(preds)

    for i, cutoff in enumerate(cutoffs):
        metrics[f"MRR@{cutoff}"] = mrrs[i]
        metrics[f"Recall@{cutoff}"] = recalls[i]
        metrics[f"Success@{cutoff}"] = successes[i]
        metrics[f"NDCG@{cutoff}"] = ndcgs[i]

    return metrics


def main():
    cutoffs = [1,3,5,10,100]

    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    eval_data_path = args.eval_data_root + args.eval_data_name.replace('/', "_") + ".jsonl"
    eval_data_corpus_path = args.eval_data_root + args.eval_data_name.replace('/', "_") + "_corpus.jsonl"

    eval_data = load_dataset("json", data_files=eval_data_path)['train']
    corpus = load_dataset("json", data_files=eval_data_corpus_path)['train']

    model = FlagModel(
        args.encoder, 
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: " if args.add_instruction else None,
        use_fp16=args.fp16
    )
    
    faiss_index = index(
        model=model, 
        corpus=corpus, 
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )
    
    scores, indices = search(
        model=model, 
        queries=eval_data, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    
    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        # indice = indice[indice != -1].tolist()
        retrieval_results.append(corpus[indice]["text"])

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["pos"])
        
    from FlagEmbedding.llm_embedder.src.utils import save_json

    metrics = evaluate(retrieval_results, ground_truths, scores, cutoffs=cutoffs)

    print(metrics)
    print('\n'.join([str(k) for k in metrics.keys()]))
    print('\n'.join([str(v) for v in metrics.values()]))
    print(args.encoder)


if __name__ == "__main__":
    main()

"""
python -m FlagEmbedding.baai_general_embedding.finetune.eval_irdatasets \
--encoder /media/code/FlagEmbedding/outputs/fiqa_HN_m3_20e_dense_64bs_1e-5 \
--fp16 \
--add_instruction \
--k 100 \
--max_passage_length 300 \
--eval_data_name lotte_science_test_forum 

python -m FlagEmbedding.baai_general_embedding.finetune.eval_irdatasets \
--encoder BAAI/bge-large-en-v1.5 \
--fp16 \
--add_instruction \
--k 100 \
--max_passage_length 512 \
--eval_data_name beir_fiqa_test

python -m FlagEmbedding.baai_general_embedding.finetune.eval_irdatasets \
--encoder /media/code/FlagEmbedding/checkpoints/fiqa_HN_dense_BAAI/bge-m3_lr1e-6_2e_t0.02 \
--fp16 \
--add_instruction \
--k 100 \
--max_passage_length 512 \
--eval_data_name beir_fiqa_test
"""
