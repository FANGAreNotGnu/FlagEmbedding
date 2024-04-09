import matplotlib.pyplot as plt


iters = [100, 200, 400, 600, 800, 1000]
pretrain = 0.21942
raw = [0.23448, 0.23499, 0.2372, 0.23357, 0.23311, 0.23291]
ep050 = [0.23811, 0.24015, 0.24194, 0.23921, 0.23966, 0.24095]
ran050 = [0.23402, 0.23697, 0.23833, 0.2347, 0.23534, 0.2365]
ep025 = [0.23742, 0.24128, 0.24396, 0.2433, 0.24195, 0.24225]
ran025 = [0.23403, 0.23648, 0.23666, 0.23304, 0.23303, 0.23322]
ep010 = [0.2351, 0.23814, 0.24085, 0.24124, 0.24058, 0.24028]
ran010 = [0.23316, 0.23495, 0.23377, 0.2307, 0.23038, 0.2302]


plt.axhline(y=pretrain, linestyle="--", color="gray", label="Pretrained")

plt.plot(iters, raw, color="blue", label="Raw Data")
plt.axhline(y=max(raw), linestyle="-.", color="blue", label="Raw Best")

plt.plot(iters, ep050, color="#c16200", label="Pruned (EP=0.50)")
plt.plot(iters, ep025, color="#881600", label="Pruned (EP=0.25)")
plt.plot(iters, ep010, color="#4e0100", label="Pruned (EP=0.10)")

plt.plot(iters, ran050, color="#7e809b", label="Pruned (RAN=0.50)")
plt.plot(iters, ran025, color="#519e8a", label="Pruned (RAN=0.25)")
plt.plot(iters, ran010, color="#476a6f", label="Pruned (RAN=0.10)")

plt.xlabel("#iterations")
plt.ylabel("NDCG@10")
plt.legend(loc="best")

plt.savefig('/media/code/FlagEmbedding/tools/vis_pruning_ablation.png')
