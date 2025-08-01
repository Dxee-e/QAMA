from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

data = np.load('space.npz')
E_base = data['E_base'] # 8
E_mutate = data['E_mutate'] # 8 64
E_mutate = E_mutate.reshape(8, 8, 8)
# print(E_base.shape, E_mutate.shape)

for i in range(8):
    x = np.arange(0, 8)
    y = np.arange(0, 8)
    x, y = np.meshgrid(x, y)
    z = E_mutate[i, :, :]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(x,y,z,cmap='viridis')
    plane = ax.plot_surface(x, y, np.ones_like(z)*E_base[i], color='r')
    fig.colorbar(surf)
    plt.savefig(f'img_space/img_space_{i}.png')