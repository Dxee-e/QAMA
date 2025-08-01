import numpy as np

seed = 42
np.random.seed(seed)

idx_range = (0, 24999)
num_samples = 40

random_indices = np.random.choice(
    range(idx_range[0], idx_range[1] + 1),
    size=num_samples,
    replace=False
)
print(random_indices)
np.save('random_indices.npy', random_indices)
