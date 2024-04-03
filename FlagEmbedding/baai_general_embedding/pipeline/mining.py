import argparse
import json
import os.path
import random
import numpy as np
import faiss
from tqdm import tqdm

from FlagEmbedding import FlagModel
from FlagEmbedding.baai_general_embedding.pipeline.utils import load_config, get_corpus_name, get_deduplicated_dataset, get_mined_dataset, seed_everything


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-base-en", type=str)
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--candidate_pool', default=None, type=str)
    parser.add_argument('--range_for_sampling', default="10-210", type=str, help="range to sample negatives")
    parser.add_argument('--use_gpu_for_searching', action='store_true', help='use faiss-gpu')
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')
    parser.add_argument('--query_instruction_for_retrieval', default="")
    parser.add_argument('--postfix', default="_minedHN")

    return parser.parse_args()


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def find_hard_neg(model, input_file, candidate_pool, output_file, sample_range, negative_number, use_gpu):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus.extend(line['pos'])
        if 'neg' in line:
            corpus.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus, batch_size=256)
    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries, batch_size=256)

    print('create index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])
    assert len(all_inxs) == len(train_data)

    for i, data in enumerate(train_data):
        query = data['query']
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            if corpus[inx] not in data['pos'] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        data['neg'] = [corpus[inx] for inx in filtered_inx]

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, 'w') as f:
        for data in train_data:
            if len(data['neg']) < negative_number:
                data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def find_easy_neg(input_file, candidate_pool, output_file, negative_number):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus.extend(line['pos'])
        if 'neg' in line:
            corpus.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, 'w') as f:
        for data in train_data:
            data['neg'] = random.sample(corpus, negative_number)
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def mining(config):

    mining_model = config.mining.model
    if mining_model is None:
        mining_model = config.pretrain.model

    model = FlagModel(mining_model, query_instruction_for_retrieval=config.data.query_instruction_for_retrieval)

    corpus_path = get_corpus_name(config)

    input_file = get_deduplicated_dataset(config)
    output_file = get_mined_dataset(config)

    if config.mining.mode == "hard":
        sample_range = config.mining.range_for_sampling.split('-')
        sample_range = [int(x) for x in sample_range]
        find_hard_neg(model,
                     input_file=input_file,
                     candidate_pool=corpus_path,
                     output_file=output_file,
                     sample_range=sample_range,
                     negative_number=config.mining.negative_number,
                     use_gpu=False)
    elif config.mining.mode == "easy":
        find_easy_neg(
            input_file=input_file,
            candidate_pool=corpus_path,
            output_file=output_file,
            negative_number=config.mining.negative_number,
        )
    else:
        raise ValueError(f"Unsupported mining mode: {config.mining.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()
    config = load_config(args.cfg)

    seed_everything(config)

    mining(config)
