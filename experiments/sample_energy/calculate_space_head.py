from QAMA.QUBO import QUBO
import torch
import numpy as np

qubo = QUBO(enable_solvers='c_sa')
device = 'cuda'

J = torch.from_numpy(np.load('J.npy')).to(device)
h = torch.from_numpy(np.load('h.npy')).to(device)

Q = qubo.prepare_coefficients(J, h).squeeze(0) # 512 512
Q = Q.reshape(8, 64, 8, 64)

x = torch.from_numpy(np.load('x.npy')).squeeze(0).to(device) # 8 64
# print(Q.shape, x.shape)

E_base_all, E_mutates_all = [], []
for head in range(8):
    Q_head = Q[head, :, head, :]
    x_head = x[head, :].unsqueeze(0)
    
    E_base = x_head @ Q_head @ x_head.T
    E_mutates = []
    for j in range(64):
        x_head_copy = x_head.clone()
        x_head_copy[:, j] = 1 - x_head_copy[:, j]
        E_mutate = x_head_copy @ Q_head @ x_head_copy.T
        E_mutates.append(E_mutate)
    E_mutates = torch.stack(E_mutates, dim=0)
    
    E_base = E_base.detach().cpu().numpy()
    E_mutates = E_mutates.detach().cpu().numpy()

    E_base_all.append(E_base.squeeze(-1).squeeze(-1))
    E_mutates_all.append(E_mutates.squeeze(-1).squeeze(-1))
E_base_all = np.stack(E_base_all, axis=0)
E_mutates_all = np.stack(E_mutates_all, axis=0)
np.savez('space.npz', E_base=E_base_all, E_mutate=E_mutates_all)