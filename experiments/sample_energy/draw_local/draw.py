import numpy as np
from matplotlib import pyplot as plt
from einops import rearrange
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['pdf.fonttype'] = 42  # 确保导出 PDF 时使用 TrueType 字体
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

x = np.load('Hd.npy').squeeze(0)
x = rearrange(x, '(h w) (n d)->n h w d',h=8, w=8, n=8, d=8)
print(x.shape)

x_mean = x.mean(axis=-1)
x_non_value_mask = (x_mean == 0.0)
# x_mean = (x_mean - x_mean.min(axis=(1,2), keepdims=True)) / (x_mean.max(axis=(1,2), keepdims=True) - x_mean.min(axis=(1,2), keepdims=True) + 1e-8)

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.1])

padding = 0.1
curve_color = '#fc9527'
curve_linewidth = 0.8

axes = []
for i in range(8):
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    axes.append(ax)
    
    
    x_norm = x_mean[i, :, :]
    x_mask = x_non_value_mask[i, :, :]
    
    if np.all(x_mask):
        x_norm = np.zeros_like(x_norm)
    else:
        values = x_norm[~x_mask]
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
        x_norm[~x_mask] = values_norm
    
    im = ax.imshow(x_norm, cmap='GnBu', interpolation='nearest')
    ax.set_title(f'Head {i + 1}', fontsize=14)
    ax.axis('off')
    
    for h in range(8):
        for w in range(8):
            rect = plt.Rectangle((w-0.5, h-0.5), 1, 1, linewidth=0.5, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
    rect = plt.Rectangle((-0.5, -0.5), 8, 8, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    for h in range(8):
        for w in range(8):
            dist_data = x[i, h, w, :]
            if np.sum(dist_data) == 0:
                continue
            x_curve = np.arange(dist_data.shape[0])
            x_norm = x_curve / (x_curve.max() or 1) # 归一化到 [0, 1]
            x_scaled = (w - 0.5 + padding) + x_norm * (1 - 2 * padding)
            
            y_min, y_max = dist_data.min(), dist_data.max()
            y_range = y_max - y_min
            y_norm = (dist_data - y_min) / (y_range if y_range > 1e-8 else 1)
            
            y_scaled = (h - 0.5 + padding) + y_norm * (1 - 2 * padding)
            
            ax.plot(x_scaled, y_scaled, color=curve_color, linewidth=curve_linewidth)

cax = fig.add_subplot(gs[:, 4])
cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.ax.set_ylabel('Norm. Energy Distribution Intensity', rotation=270, labelpad=10, fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0.1)

# plt.tight_layout()
plt.savefig('energy.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.savefig('energy.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
# plt.show()