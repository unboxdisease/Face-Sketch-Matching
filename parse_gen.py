import numpy as np
f = 'Assets2/similarity_synthetic.txt'

lines = [line.strip() for line in open(f).readlines()]
genuine_scores = []
for line in lines:
    path, path2, score = line.split(",")[:3]
    # print(('/research/prip-jainkush/Datasets/ForensicSketches/FullDataset/sketch/'+path[:-1] , '/research/prip-jainkush/Datasets/ForensicSketches/FullDataset/sketch/'+path2 ,score))
    if path.split('_')[0] != path2.split('_')[0]:
        print((path,path2,score))
        genuine_scores.append((path,path2,score))
print(len(genuine_scores))
np.save('Assets2/ROC_Synthetic/imposter_scores_.npy',np.array(genuine_scores))
