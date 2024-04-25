import torch
import numpy as np
import matplotlib.pyplot as plt

import os


root_dir = "/media/code/FlagEmbedding/visualization/nfcorpus_scores"

score_files = os.listdir(os.path.join(root_dir, "scores"))
scores_dict = {}
for file in score_files:
    iter = int(os.path.basename(file)[:-3])  # xxx.pt
    scores_dict[iter] = torch.load(os.path.join(root_dir, "scores", file)).numpy()

iters = np.sort(list(scores_dict.keys()))
N = len(iters)
D = len(scores_dict[iters[0]])
print(f"#scores: {N}")
print(f"score dim: {D}")
scores = np.zeros((N, D))

for i, iter in enumerate(iters):
    scores[i] = scores_dict[iter]

#rint(scores[:, 1])

plt.imshow(scores.transpose((1,0)), cmap="jet")
plt.colorbar()
plt.savefig(os.path.join(root_dir, "scores.png"))
