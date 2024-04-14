import matplotlib.pyplot as plt
import numpy as np

X = ['FEVER','NFCorpus','TripClick'] 
pretrain = [0.86779, 0.45101, 0.21942]
finetune = [0.87635, 0.47308, 0.2372]
finetune_pruned = [0.90531, 0.48972, 0.24396]

f_p = []  # (finetune - pretrain)/pretrain
fp_p = []  # (finetune_pruned - pretrain)/pretrain
for i in range(len(pretrain)):
    f_p.append((finetune[i] - pretrain[i])/pretrain[i]*100)
    fp_p.append((finetune_pruned[i] - pretrain[i])/pretrain[i]*100)
  
X_axis = np.arange(len(X)) 
  
#plt.bar(X_axis - 0.2, pretrain, 0.2, label = 'Pretrain performance') 
#plt.bar(X_axis, finetune, 0.2, label = 'Finetune (No pruning) performance') 
#plt.bar(X_axis + 0.2, finetune_pruned, 0.2, label = 'Finetuned (Pruned, EP=0.25) performance') 

plt.bar(X_axis - 0.2, f_p, 0.4, label = 'Finetuning (No pruning) improvement') 
plt.bar(X_axis + 0.2, fp_p, 0.4, label = 'Finetuning (Pruned, EP=0.25) improvement') 

  
plt.xticks(X_axis, X) 
plt.xlabel("Datasets") 
plt.ylabel("Performance improvements (%)") 
plt.title("Performance improvements (over pretrained model) on each dataset.") 
plt.legend()

plt.savefig('/media/code/FlagEmbedding/tools/visualization/dataset_bar.png')
