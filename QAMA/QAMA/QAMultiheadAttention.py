import numpy as np
import torch
from icecream import ic
from torch import Tensor, nn
from torch.nn import functional as F
from QAMA.QUBO import QUBO
from typing import Optional, Union
from einops import rearrange
import line_profiler

EPS = 1e-6

@torch._dynamo.disable
class QAMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        embed_dim: int, # dim_head
        num_heads: int,
        args_model: dict = {},
        args_solver: dict = {},
        enable_solvers: Union[list, str] = "all",
    ):
        """Args:

        Arguments for Multi-head Attention:
        - num_heads: int, number of heads

        Arguments for QUBO Model:


        Arguments for Quantum Annealing Solver:
        solver_backend: str, the backend of the solver, support list -> ['kaiwu_sa', 'gurobi', 'kaiwu_cim', 'montecarlo_greedy']

        """
        super(QAMultiheadAttention, self).__init__()
        self.qubo = QUBO(
            model_args=args_model,
            solver_args=args_solver,
            enable_solvers=enable_solvers,
        )
        
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        
        self.norm_h = nn.LayerNorm(embed_dim)
        
        head_dim = d_model // num_heads     
        self.scale = head_dim ** -0.5      
        inner_dim = embed_dim * num_heads
        self.Wq = nn.Linear(d_model, inner_dim, bias=False)
        self.Wk = nn.Linear(d_model, inner_dim, bias=False)
        self.Wv = nn.Linear(d_model, inner_dim, bias=False)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        
        self.E2V = nn.Parameter(torch.randn(embed_dim), requires_grad=True)

    @line_profiler.profile
    def forward(self, Q: Tensor, K: Optional[Tensor]=None, V: Optional[Tensor]=None, is_return_xJhH: bool = False, solver_name: str = "c_sa"):
        # QKV: [batch_size, seq_length, d_model]
        if K is None:
            K = Q
        if V is None:
            V = Q
        Q = self.Wq(self.norm_q(Q))
        K = self.Wk(self.norm_k(K))
        V = self.Wv(self.norm_v(V))
        Q = rearrange(Q, "b s (n d) -> b n s d", n=self.num_heads, d=self.head_dim)
        K = rearrange(K, "b s (n d) -> b n s d", n=self.num_heads, d=self.head_dim)
        V = rearrange(V, "b s (n d) -> b n s d", n=self.num_heads, d=self.head_dim)
        
        h = (V * self.E2V).mean(dim=-1)

        J = (Q @ K.transpose(-1, -2)) * self.scale
        J = (J + J.transpose(-1, -2)) / 2.0  # make J symmetric
        J[:, :, torch.eye(Q.shape[2], dtype=bool)[:, :]] = 0.0

        # solve
        x = self.qubo.solve(J=J, h=h, solver_name=solver_name)
        H = self.calculate_energy(J, h, x.detach())

        Hd = H.unsqueeze(-1) * self.E2V
        Hd = rearrange(Hd, "b n s d -> b s (n d)")

        if is_return_xJhH:
            return x, J, h, H, Hd
        # np.save('./sample_energy/x.npy', x.cpu().numpy())
        # np.save('./sample_energy/J.npy', J.cpu().numpy())
        # np.save('./sample_energy/h.npy', h.cpu().numpy())
        # np.save('./sample_energy/H.npy', H.cpu().numpy())
        # np.save('./sample_energy/Hd.npy', Hd.cpu().numpy())
        return Hd

    def calculate_energy(self, J: Tensor, h: Tensor, x: Tensor) -> Tensor:
        """Calculate the energy of the QUBO model."""
        b, t, n = x.shape
        J_objective = (x.unsqueeze(3) * x.unsqueeze(2) * J).sum(dim=-1)
        h_objective = h * x
        energy = J_objective + h_objective
        return -energy

if __name__ == "__main__":
    # Test the QAMultiheadAttention class
    batch_size = 128
    seq_length = 49
    d_model = 256
    num_heads = 8
    embed_dim = 32

    Q = torch.randn(batch_size, seq_length, d_model).to('cuda')
    K = torch.randn(batch_size, seq_length, d_model).to('cuda')
    V = torch.randn(batch_size, seq_length, d_model).to('cuda')

    model = QAMultiheadAttention(d_model, embed_dim, num_heads, enable_solvers=['gurobi', 'c_sa', 'montecarlo_greedy']).to('cuda')
    import time
    start_time = time.time()
    output = model(Q, K, V, solver_name='montecarlo_greedy')
    print(f"Time taken: {time.time() - start_time:.4f} seconds")
    print(output.shape) 
    loss = torch.mean(output)
    start_time = time.time()
    loss.backward()
    print(f"Time taken for backward: {time.time() - start_time:.4f} seconds")
    
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     output = model(Q, K, V, solver_name='montecarlo_greedy')
    #     loss = torch.mean(output)
    #     loss.backward()
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    