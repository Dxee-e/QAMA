import numpy as np
from matplotlib import pyplot as plt

x = np.load('x.npy').squeeze(0)
x = x.reshape(8, 8, 8)

for i in range(8):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(x[i, :, :])
    plt.colorbar()
    plt.savefig(f'img_x/img_x_{i}.png')