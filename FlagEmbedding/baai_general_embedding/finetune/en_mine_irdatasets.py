import argparse
import json
import os.path
import random
import numpy as np
import faiss
from tqdm import tqdm

from FlagEmbedding import FlagModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--candidate_pool', default=None, type=str)
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')
    parser.add_argument('--postfix', default="_minedEN")

    return parser.parse_args()


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def find_knn_neg(input_file, candidate_pool, output_file, negative_number):
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


if __name__ == '__main__':
    args = get_args()
    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    input_file = f"/media/data/flagfmt/{args.dataset_name.replace('/','_')}.jsonl"
    output_file = f"/media/data/flagfmt/{args.dataset_name.replace('/','_')}{args.postfix}/{args.dataset_name.replace('/','_')}{args.postfix}.jsonl"

    find_knn_neg(
        input_file=input_file,
        candidate_pool=args.candidate_pool,
        output_file=output_file,
        negative_number=args.negative_number,
    )
