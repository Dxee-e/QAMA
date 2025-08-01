import numpy as np
import torch
from torch import Tensor, nn
import importlib.util
import os
import line_profiler

# hybird solver of Monte Carlo Greedy and C-Simulated Annealing
class Solver:
    def __init__(self, args_solver: dict):
        self.args_solver = args_solver
        assert 'montecarlo_greedy' in self.args_solver, "Please provide 'montecarlo_greedy' solver arguments."
        assert 'c_sa' in self.args_solver, "Please provide 'c_sa' solver arguments."

        from QAMA.backend_montecarlo_greedy.solver import Solver as MonteCarloGreedySolver
        from QAMA.backend_c_sa.solver import Solver as CSA_Solver
        self.montecarlo_greedy_solver = MonteCarloGreedySolver(args_solver=self.args_solver['montecarlo_greedy'])
        self.csa = CSA_Solver(args_solver=self.args_solver['c_sa'])

    @torch.no_grad
    def solve(self, Q: Tensor) -> Tensor:
        mt_x = self.montecarlo_greedy_solver.solve(Q)
        mt_energy = (mt_x.unsqueeze(-1) * mt_x.unsqueeze(-2) * Q).sum(dim=(-1, -2))
        
        csa_x = self.csa.solve(Q)
        csa_energy = (csa_x.unsqueeze(-1) * csa_x.unsqueeze(-2) * Q).sum(dim=(-1, -2))
        
        accept_mask = csa_energy < mt_energy
        x = torch.where(accept_mask.unsqueeze(-1), csa_x, mt_x)
        return x



if __name__ == "__main__":
    device = 'cuda'
    torch.manual_seed(0)
    batch_size = 128
    qubit_num = 8*49
    Q = torch.randn(batch_size, qubit_num, qubit_num, device=device, dtype=torch.float32)
    Q = (Q + Q.transpose(-1, -2)) / 2 
    solver = Solver({
        'c_sa': {
            "initial_temperature": 1000.0,
            "alpha": 0.99,
            "cutoff_temperature": 0.001,
            "iterations_per_t": 5,
            "patience": 10,
            "is_local_search": True,
            "flag_evolution_history": False,
            "history_file_path": "",
            "rand_seed": None,
            "is_input_check": False,
        },
        'montecarlo_greedy': {
            "sample_num": 200,
            "seed": 0,
            "device": "cuda",
            "iterations": 3,
        },
        'hybird': {
            # None means using the default parameters of the montecarlo_greedy and c_sa solvers
        }
    })
    import time

    start = time.time()
    x = solver.solve(Q)
    print("Time taken:", time.time() - start)
    print((x == 1).sum(), (x == 0).sum())
    print(x.shape)
