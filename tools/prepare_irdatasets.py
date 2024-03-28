import json
import numpy as np
import os
os.environ["IR_DATASETS_HOME"] = "/media/data/ir_datasets/"
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tqdm import tqdm
from typing import List

import ir_datasets


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


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    for dataset_name in args.dataset_names:
        avg_num_pos, std_num_pos, confident_max_num_pos = prepare_data(dataset_name=dataset_name)
        print(f"For HN mining, avg_num_pos: {avg_num_pos}")
        print(f"For HN mining, confident_max_num_pos: {confident_max_num_pos}")


if __name__ == "__main__":
    main()


"""
Next step: Hard Negative Mining (https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-m3 \
--input_file /media/data/flagfmt/beir_fiqa_train.jsonl \
--output_file /media/data/flagfmt/beir_fiqa_train_minedHN/beir_fiqa_train_minedHN.jsonl \
--range_for_sampling 10-100 \
--negative_number 10 \
--use_gpu_for_searching 

Then train:
Use bge-m3 self-distill/deepspeed?

torchrun --nproc_per_node 8 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir /media/code/FlagEmbedding/outputs/fiqa_HN_m3_3e_dense_64bs_1e-5 \
--model_name_or_path BAAI/bge-m3 \
--train_data /media/data/flagfmt/beir_fiqa_train_minedHN \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 3 \
--per_device_train_batch_size 8 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 512 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" 

And evaluate:
python -m FlagEmbedding.baai_general_embedding.finetune.eval_irdatasets \
--encoder /media/code/FlagEmbedding/outputs/fiqa_HN_m3_3e_dense_64bs_1e-5 \
--fp16 \
--add_instruction \
--k 100 \
--max_passage_length 512 \
--eval_data_name beir_fiqa_test 


python -m FlagEmbedding.baai_general_embedding.finetune.eval_irdatasets \
--encoder BAAI/bge-m3 \
--fp16 \
--add_instruction \
--k 100 \
--max_passage_length 512 \
--eval_data_name beir_fiqa_test 

python -m FlagEmbedding.baai_general_embedding.finetune.eval_irdatasets \
--encoder BAAI/bge-m3 \
--fp16 \
--add_instruction \
--k 100 \
--max_passage_length 512 \
--eval_data_name beir/nfcorpus/test
"""
