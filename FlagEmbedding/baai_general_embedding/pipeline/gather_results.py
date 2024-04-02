import os
import numpy as np

from FlagEmbedding.llm_embedder.src.utils import load_json
from FlagEmbedding.baai_general_embedding.pipeline.utils import get_result_path


model_save_path = "/media/data/ragckpt/tripclick_train_head/dedup_BAAI_bge-large-en-v1.5_kept0.25_ep-mined_hard_100-1100_500-1e-06_32e-0.02T_0.0wd"

metric_name = "ndcg@10"

def get_ckpt_iter(ckpt):
    return int(ckpt[11:])


ckpts = os.listdir(model_save_path)
ckpts = filter(lambda x: x[:11]=="checkpoint-", ckpts)

ckpt_iters = [get_ckpt_iter(ckpt) for ckpt in ckpts]
sorted_idx = np.argsort(ckpt_iters)


for i in sorted_idx:
    print(get_ckpt_iter(ckpt_iters[i]))

for i in sorted_idx:
    ckpt = ckpts[i]
    result_path = get_result_path(ckpt, None)
    metrics = load_json(result_path)
    print(metrics[metric_name])
