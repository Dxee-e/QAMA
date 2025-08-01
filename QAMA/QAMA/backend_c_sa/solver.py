import numpy as np
import torch
from torch import Tensor, nn
import importlib.util
import os
import line_profiler


class Solver:
    def __init__(self, args_solver: dict):
        self.args_solver = args_solver

        # from simulated_annealing_qubo import simulated_annealing_qubo
        spec = importlib.util.spec_from_file_location('simulated_annealing_qubo', os.path.join(os.path.dirname(__file__), "simulated_annealing_qubo.so"))
        lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib_module)
        self.csa = lib_module.simulated_annealing_qubo

    @torch.no_grad
    def solve(self, Q: Tensor, csa_init_x: Tensor = None) -> Tensor:
        x = self.csa(
            Q=Q,
            initial_temperature=self.args_solver["initial_temperature"],
            alpha=self.args_solver["alpha"],
            cutoff_temperature=self.args_solver["cutoff_temperature"],
            iterations_per_t=self.args_solver["iterations_per_t"],
            patience=self.args_solver["patience"],
            is_local_search=self.args_solver["is_local_search"],
            flag_evolution_history=self.args_solver["flag_evolution_history"],
            history_file_path=self.args_solver["history_file_path"],
            rand_seed=self.args_solver["rand_seed"],
            is_input_check=self.args_solver["is_input_check"],
            csa_init_x=csa_init_x,
        )
        return x


if __name__ == "__main__":
    device = 'cuda'
    torch.manual_seed(0)
    batch_size = 1
    qubit_num = 8*49
    Q = torch.randn(batch_size, qubit_num, qubit_num, device=device, dtype=torch.float32)
    Q = (Q + Q.transpose(-1, -2)) / 2 
    solver = Solver({
        "initial_temperature": 1000.0,
        "alpha": 0.99,
        "cutoff_temperature": 0.001,
        "iterations_per_t": 5,
        "patience": 50,
        "is_local_search": True,
        "flag_evolution_history": False,
        "history_file_path": "",
        "rand_seed": None,
        "is_input_check": False,
    })
    import time

    start = time.time()
    x = solver.solve(Q)
    print("Time taken:", time.time() - start)
    print((x == 1).sum(), (x == 0).sum())
    print(x.shape)
