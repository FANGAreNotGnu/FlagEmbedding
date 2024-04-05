import matplotlib.pyplot as plt


iters = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
pretrain = 0.45141
raw = [0.46582, 0.4657, 0.46772, 0.46902, 0.4701, 0.47308, 0.47003, 0.47081]
ep050 = [0.46424, 0.47482, 0.47506, 0.47939, 0.48437, 0.48322, 0.48487, 0.48747]
ep025 = [0.47294, 0.475, 0.47891, 0.47851, 0.47874, 0.48269, 0.48382, 0.48512]


plt.axhline(y=pretrain, linestyle="--", color="gray", label="Pretrained")

plt.plot(iters, raw, color="blue", label="Raw Data")
plt.axhline(y=max(raw), linestyle="-.", color="blue", label="Raw Best")

plt.plot(iters, ep050, label="Pruned (EP=0.50)")
plt.plot(iters, ep025, label="Pruned (EP=0.25)")

plt.xlabel("#iterations")
plt.ylabel("NDCG@10")
plt.legend(loc="best")

plt.savefig('/media/code/FlagEmbedding/tools/vis_pruning_ablation.png')
