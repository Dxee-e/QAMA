from QAMA.QUBO import QUBO
import torch
import numpy as np

qubo = QUBO(enable_solvers='c_sa')
device = 'cuda'

J = torch.from_numpy(np.load('J.npy')).to(device)
h = torch.from_numpy(np.load('h.npy')).to(device)

Q = qubo.prepare_coefficients(J, h).squeeze(0) # 512 512

x = torch.from_numpy(np.load('x.npy')).squeeze(0).to(device) # 8 64
x = x.view(-1).unsqueeze(0)
# print(Q.shape, x.shape)

E_base = x @ Q @ x.T
E_mutates = []
for j in range(512):
    x_copy = x.clone()
    x_copy[:, j] = 1 - x_copy[:, j]
    E_mutate = x_copy @ Q @ x_copy.T
    E_mutates.append(E_mutate)
E_mutates = torch.stack(E_mutates, dim=0)

E_base = E_base.detach().cpu().numpy().squeeze(-1).squeeze(-1)
E_mutates = E_mutates.detach().cpu().numpy().squeeze(-1).squeeze(-1)

np.savez('space_all.npz', E_base=E_base, E_mutate=E_mutates)