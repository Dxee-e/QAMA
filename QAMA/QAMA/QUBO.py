import torch
from torch import Tensor
import numpy as np
from typing import Union
import line_profiler

DEFAULT_MODEL_ARGS = {
    # QUBO model
    "quadratic_coeffecient": 1.0,
    "linear_fix_coeffecient": 0.16,
    "penalty_multi_head_fix_coeffecient": 0.8,
}

DEFAULT_SOLVER_ARGS = {
    'c_sa': {
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
    },
    'gurobi': {
        # parallel
        "batch_num_process": 8,
        "timeout": 10,
        # gurobi solver parameters
        "mipgap": 0.01,
        "gurobi_num_threads": 1,
    },
    'kaiwu_sa': {
        # kaiwu SDK init license
        "user_id": '69878024601862146',
        "sdk_code": '0i4T6LY1XygfwN3MWa8Fjq27OaT0sq',
        # parallel
        "batch_num_process": 8,
        "timeout": 10,
        # Simulated Annealing Parameters
        "initial_temperature": 100,
        "alpha": 0.99,
        "cutoff_temperature": 1e-3,
        "iterations_per_t": 10,
        "size_limit": 10,
        "rand_seed": None,
        "sa_num_process": 1,
    },
    'montecarlo_greedy': {
        "sample_num": 200,
        "seed": 0,
        "device": "cuda",
        "iterations": 3,
    },
    'hybird': {},
    'cim': {
        # kaiwu SDK init license
        "user_id": '69878024601862146',
        "sdk_code": '0i4T6LY1XygfwN3MWa8Fjq27OaT0sq',
        # stage
        'stage': 'generate', # 'generate' or 'solve'
        'batch_idx': [],
    }
}

class QUBO:
    def __init__(self,model_args: dict={}, solver_args: dict = {}, enable_solvers: Union[list, str] = 'all'):
        self.args_model = DEFAULT_MODEL_ARGS.copy()
        self.args_solver = DEFAULT_SOLVER_ARGS.copy()
        
        for key, value in model_args.items():
            if key in self.args_model:
                self.args_model[key] = value
            else:
                raise ValueError(f"Unknown model argument: {key}")
            
        for key, value in solver_args.items():
            if key not in self.args_solver:
                raise ValueError(f"Unknown solver: {key}")
            for key_solver, value_solver in value.items():
                if key_solver in self.args_solver[key]:
                    self.args_solver[key][key_solver] = value_solver
                else:
                    raise ValueError(f"Unknown solver argument: {key}.{key_solver}")
        
        
        if type(enable_solvers) == str:
            if enable_solvers == 'all':
                enable_solvers = list(self.args_solver.keys())
            else:
                enable_solvers = [enable_solvers]
        if len(enable_solvers) == 0:
            raise ValueError("No solver is enabled. Please specify at least one solver.")
        
        if 'hybird' in enable_solvers:
            if 'montecarlo_greedy' not in self.args_solver or 'c_sa' not in self.args_solver:
                raise ValueError("Hybird solver requires both 'montecarlo_greedy' and 'c_sa' solvers to be defined in args_solver.")
            self.args_solver['hybird'].update({
                "montecarlo_greedy": self.args_solver['montecarlo_greedy'],
                "c_sa": self.args_solver['c_sa']
            })
        
        self.solvers = {}
        for solver_name in enable_solvers:
            if solver_name not in self.args_solver:
                raise ValueError(f"Solver {solver_name} is not defined in args_solver.")
            if solver_name not in self.solvers:
                # Import the solver dynamically based on the name
                module_name = f"QAMA.backend_{solver_name}.solver"
                module = __import__(module_name, fromlist=['Solver'])
                SolverClass = getattr(module, 'Solver')
                self.solvers[solver_name] = SolverClass(args_solver=self.args_solver[solver_name])
    
    def prepare_coefficients(self, J: Tensor, h: Tensor) -> Tensor:
        # J: [b, t, n, n]
        # h: [b, t, n]
        assert (J==J.transpose(-1, -2)).all(), "J must be symmetric"
        assert J[:, :, torch.eye(J.shape[2], dtype=bool)[:, :]].sum() == 0, "J must have zero diagonal elements"
        b, t, n, _ = J.shape
        device = J.device
        dtype = J.dtype
        Q = torch.zeros((b, t, n, t, n), device=device, dtype=dtype)
        
        t_idxs = torch.arange(t, device=device)
        n_idxs = torch.arange(n, device=device)
        
        t_indices = t_idxs.view(-1, 1, 1)
        j_indices = n_idxs.view(1, -1, 1)
        k_indices = n_idxs.view(1, 1, -1)
        Q[:, t_indices, j_indices, t_indices, k_indices] -= J * self.args_model["quadratic_coeffecient"] 
        Q[:, t_indices, j_indices, t_indices, j_indices] -= h[:, t_indices, j_indices] * self.args_model["linear_fix_coeffecient"] * n

        if t > 1:
            k_indices = n_idxs.view(-1, 1, 1)
            t0_indices = t_idxs.view(1, -1, 1)
            t1_indices = t_idxs.view(1, 1, -1)
            penalty_value = self.args_model["penalty_multi_head_fix_coeffecient"] / 2 * (np.sqrt(2 / np.pi) * n / (t - 1)) 
            Q[:, t0_indices, k_indices, t1_indices, k_indices] += penalty_value
            Q[:, t0_indices, k_indices, t0_indices, k_indices] -= penalty_value

        Q = Q.view(b, t*n, t*n)
        return Q
        
        
    @line_profiler.profile
    def solve(self, J: Tensor, h: Tensor, solver_name: str) -> Tensor:
        """
        Solve the QUBO problem given the interaction matrix J and linear coefficients h.
        
        Args:
            J (torch.Tensor): Interaction matrix of shape (b, t, n, n).
            h (torch.Tensor): Linear coefficients of shape (b, t, n).
        
        Returns:
            torch.Tensor: Solution of shape (b, t, n).
        """
        b, t, n, _ = J.shape
        Q = self.prepare_coefficients(J, h)
        if solver_name not in self.solvers:
            raise ValueError(f"Solver {solver_name} is not defined in the solvers.")
        x = self.solvers[solver_name].solve(Q).to(torch.float32)
        return x.view(b, t, n)

    
if __name__ == "__main__":
    # Example usage
    qubo = QUBO()
    J = torch.rand((2, 3, 4, 4), dtype=torch.float32)
    J = (J + J.transpose(-1, -2)) / 2 
    J[:, :, torch.eye(J.shape[2], dtype=bool)[:, :]] = 0 
    h = torch.rand((2, 3, 4), dtype=torch.float32)
    for solver_name in qubo.solvers:
        print(f"Solving with {solver_name} solver...")
        x = qubo.solve(J, h, solver_name)
        print(f"Solution shape: {x.shape}")
        print("-" * 40)
    
    
