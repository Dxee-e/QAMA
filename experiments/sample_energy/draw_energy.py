import numpy as np
from matplotlib import pyplot as plt
from einops import rearrange

x = np.load('Hd.npy').squeeze(0)
x = rearrange(x, '(h w) (n d)->n h w d',h=8, w=8, n=8, d=8)
print(x.shape)

for i in range(8):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(x[i, :, :, :].mean(axis=-1))
    plt.colorbar()
    plt.savefig(f'img_energy/img_energy_{i}.png')