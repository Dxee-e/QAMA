from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

x = np.load('x.npy').squeeze(0).squeeze(0)
H = np.load('H.npy').squeeze(0).squeeze(0)
Hd = np.load('Hd.npy').squeeze(0)
print(f"x shape: {x.shape}, H shape: {H.shape}, Hd shape: {Hd.shape}")
# x shape: (512,), H shape: (512,), Hd shape: (512, 64)

print(x)
for i in range(512):
    if x[i] ==1:
        print(i , end=' ')
print('\n')

sentence = pd.read_csv('/root/workspace/project_QAMA/experiments/common/data/IMDB/test.csv').iloc[9668].sentence
print(f"Sentence: {sentence}")
print(len(sentence.split()))