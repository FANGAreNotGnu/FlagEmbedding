import json
import os
os.environ["IR_DATASETS_HOME"] = "/media/data/ir_datasets/"
from tqdm import tqdm

import ir_datasets


dataset_name = "lotte/science/test/forum"

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
    query_data[qid]["pos_did"].append(did)

with open(target_path, 'w+') as f:
    for qid, q_data in tqdm(query_data.items()):
        query_text = q_data["text"]
        pos_texts = []
        for pos_did in q_data["pos_did"]:
            doc = docstore.get(pos_did)
            pos_texts.append(doc.text)
        #print(len(pos_texts))

        f.write(json.dumps({"query": query_text, "pos": pos_texts}) + "\n")

        # TODO: print doc distribution per query to set hyperparameters for HN mining


"""
Next step: Hard Negative Mining (https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-m3 \
--input_file /media/data/flagfmt/lotte_science_dev_forum.jsonl \
--output_file /media/data/flagfmt/lotte_science_dev_forum_minedHN/lotte_science_dev_forum_minedHN.jsonl \
--range_for_sampling 30-200 \
--negative_number 15 \
--use_gpu_for_searching 

Then train:
Use bge-m3 deepspeed?

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node 6 \
-m FlagEmbedding.BGE_M3.run \
--output_dir /media/code/FlagEmbedding/outputs/lotte_science_dev_forum_minedHN_m3_20e \
--model_name_or_path BAAI/bge-m3 \
--train_data /media/data/flagfmt/lotte_science_dev_forum_minedHN \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 20 \
--per_device_train_batch_size 1 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 1024 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--same_task_within_batch True \
--unified_finetuning True \
--use_self_distill True

"""
