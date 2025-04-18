from torch import nn
from torch import Tensor
import torch
from torch.nn import functional as F
import numpy as np
from icecream import ic

EPS = 1e-6

class QAMultiheadAttention(nn.Module):
    def __init__(self, num_heads: int, args_model: dict={}, solver_backend: str='gurobi', args_solver: dict={}):
        """ Args:
        
        Arguments for Multi-head Attention: 
        - num_heads: int, number of heads
        
        Arguments for QUBO Model:
        
        
        Arguments for Quantum Annealing Solver:
        solver_backend: str, the backend of the solver, support list -> ['kaiwu_sa', 'gurobi', 'kaiwu_cim', 'gpu_best', 'cpu_best']
        
        """
        super(QAMultiheadAttention, self).__init__()

        assert num_heads > 0, "num_heads must be greater than 0"
        assert solver_backend in ['kaiwu_sa', 'gurobi', 'kaiwu_cim', 'gpu_best', 'cpu_best'], "solver_backend must be in ['kaiwu_sa', 'gurobi', 'kaiwu_cim', 'gpu_best', 'cpu_best']"
        # assert mode in ['train', 'test'], "mode must be in ['train', 'test']" 
        
        # self.mode = mode # in test mode, energy funtion is not call, which is only for backward
        
        self.solver_backend = solver_backend
        self.solver = None
        if solver_backend == 'kaiwu_sa':
            from QAMA.backend_kaiwu_sa.solver import Solver
            self.solver = Solver(args_model=args_model, args_solver=args_solver)
        else:
            raise NotImplementedError(f"Solver backend {solver_backend} is not implemented.")
            
        self.J_bn = nn.BatchNorm1d(num_heads)
        self.h_bn = nn.BatchNorm1d(num_heads)
        self.H_bn = nn.BatchNorm2d(num_heads)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        # QKV: [batch_size, num_heads, seq_len, d_k]
        
        # h - entropy
        V = V / (torch.sum(V, dim=-1, keepdim=True) + EPS)
        V = torch.clamp(V, min=EPS, max=1.0)
        hi = -V * torch.log2(V)
        h = torch.sum(hi, dim=-1)
        
        # J - JS divergence
        p_Q = F.softmax(Q, dim=-1)
        p_K = F.softmax(K, dim=-1)
        m = 0.5 * (p_Q.unsqueeze(3) + p_K.unsqueeze(2))
        kl_pQ_m = torch.sum(p_Q.unsqueeze(3) * (torch.log2(p_Q.unsqueeze(3) / (m+EPS) + EPS)),dim=-1)
        kl_pK_m = torch.sum(p_K.unsqueeze(2) * (torch.log2(p_K.unsqueeze(2) / (m+EPS) + EPS)),dim=-1)
        J = 0.5 * kl_pQ_m + 0.5 * kl_pK_m

        # norm for J
        J_nondiag_mask = ~np.eye(Q.shape[2], dtype=bool)[:,:]
        non_diagonal_elemetns = J[:, :, J_nondiag_mask]
        non_diagonal_elemetns = self.J_bn(non_diagonal_elemetns)
        J[:, :, J_nondiag_mask] = non_diagonal_elemetns
        
        # norm for h
        h = self.h_bn(h)

        # solve
        x = self.solver.solve(J.detach().cpu().numpy(), h.detach().cpu().numpy())
        # if self.mode == 'train':
        H = self.solver.energy_function(J, h, x)
        
        # output energy
        Hd = H.unsqueeze(-1) * hi / h.unsqueeze(-1)
        Hd = self.H_bn(-Hd)
        return Hd
    
