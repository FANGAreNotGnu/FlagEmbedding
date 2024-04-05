import argparse
import json
import numpy as np
import os
os.environ["IR_DATASETS_HOME"] = "/media/data/ir_datasets/"
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tqdm import tqdm
from typing import List

import ir_datasets

from FlagEmbedding.baai_general_embedding.pipeline.utils import load_config, load_positive_pairs, save_positive_pairs, get_flag_format_dataset, get_deduplicated_dataset, seed_everything


@dataclass
class Args:
    dataset_names: List[str] = field(default_factory=lambda: [])


def prepare_data(dataset_name):

    target_path = "/media/data/flagfmt/" + dataset_name.replace("/", "_") + ".jsonl"
    target_corpus_path = "/media/data/flagfmt/" + dataset_name.replace("/", "_") + "_corpus.jsonl"

    dataset = ir_datasets.load(dataset_name)
    docstore = dataset.docs_store()

    with open(target_corpus_path, 'w+') as f:
        for doc in tqdm(dataset.docs_iter()):
            f.write(json.dumps({"text": doc.text}) + "\n")

    query_data = {}
    for query in dataset.queries_iter():
        qid = query.query_id
        query_text = query.text
        query_data[qid] = {"text": query_text, "pos_did": []}

    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        did = qrel.doc_id
        rel = qrel.relevance
        # TODO: add relevance ifelse
        if int(rel) > 0:
            query_data[qid]["pos_did"].append(did)

    num_pos_samples = []
    with open(target_path, 'w+') as f:
        for qid, q_data in tqdm(query_data.items()):
            query_text = q_data["text"]
            pos_texts = []
            for pos_did in q_data["pos_did"]:
                try:
                    doc = docstore.get(pos_did)
                except KeyError:
                    print(f"doc id not found: {pos_did}")
                    continue
                pos_texts.append(doc.text)
            num_pos_samples.append(len(pos_texts))
            #print(len(pos_texts))

            f.write(json.dumps({"query": query_text, "pos": pos_texts}) + "\n")

    # Use distribution of positive sample per query to set hyperparameters for HN mining
    return np.mean(num_pos_samples), np.std(num_pos_samples), np.mean(num_pos_samples) + 3 * np.std(num_pos_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()
    config = load_config(args.cfg)

    seed_everything(config)

    for dataset_name in [config.data.train, config.data.val, config.data.test]:
        avg_num_pos, std_num_pos, confident_max_num_pos = prepare_data(dataset_name=dataset_name)
        print(dataset_name)
        print(f"For HN mining, avg_num_pos: {avg_num_pos}")
        print(f"For HN mining, confident_max_num_pos: {confident_max_num_pos}")
