"""generate Q.csv file and upload to online plantform for batch input manually"""

import kaiwu as kw
import numpy as np
import torch
import torch.multiprocessing as mp
from icecream import ic
from torch import Tensor
import os
import pandas as pd
import json
from tqdm import tqdm

class Solver:
    def __init__(self, args_solver: dict):
        self.args_solver = args_solver
        # init kaiwu SDK
        kw.license.init(self.args_solver["user_id"], self.args_solver["sdk_code"])
        self.stage = self.args_solver['stage']
        assert self.stage in ['generate', 'solve'], "stage must be 'generate' or 'solve'"
        self.batch_idx = self.args_solver['batch_idx'] 
        assert len(self.batch_idx) > 0, "batch_idx must be a non-empty list of indices to process"

    def _solve_step(self, Q: np.ndarray, inner_idx: int) -> np.ndarray:
        # no batch for parallel
        # Q: [m, m]
        m, _ = Q.shape

        x = kw.qubo.ndarray((m, ), "x", kw.qubo.Binary)
        qubo_model = kw.qubo.QuboModel()
        objective = x[:, None] * x[None, :] * Q
        objective = kw.qubo.quicksum([
            kw.qubo.quicksum([
                objective[i, j] for j in range(m)
            ]) for i in range(m)
        ])
        qubo_model.set_objective(objective)

        # solve
        if self.stage == 'generate':
            qubo_mat = qubo_model.get_qubo_matrix(bit_width=8)
            idx = self.batch_idx[inner_idx]
            pd.DataFrame(qubo_mat).to_csv(f'./cim_run/Q/Q_{idx}.csv', index=False, header=False)
        elif self.stage == 'solve':
            idx = self.batch_idx[inner_idx]
            x = json.load(open(f'./cim_run/results/solution_{idx}.log'))
            x = x[0]['solutionVector']
            x = np.array(x, dtype=np.float32)
            assert len(x) == m, f"Length of solution vector {len(x)} does not match the size of Q {m}"
            return x
        else:
            raise ValueError("stage must be 'generate' or 'solve'")

    def solve(self, Q: Tensor) -> Tensor:
        device = Q.device
        dtype = Q.dtype
        batch = Q.shape[0]
        Q = Q.detach().cpu().numpy()

        results = []
        for b in tqdm(range(batch)):
            results.append(self._solve_step(Q[b, :, :], b))
        if self.stage == 'generate':
            exit(0)
        results = np.stack(results, axis=0)
        return torch.from_numpy(results).type(dtype).to(device)
    

if __name__ == "__main__":
    device = 'cuda'
    torch.manual_seed(0)
    batch_size = 3
    qubit_num = 8*49
    Q = torch.randn(batch_size, qubit_num, qubit_num, device=device, dtype=torch.float32)
    Q = (Q + Q.transpose(-1, -2)) / 2 
    solver = Solver({
        # kaiwu SDK init license
        "user_id": "69878024601862146",
        "sdk_code": "0i4T6LY1XygfwN3MWa8Fjq27OaT0sq",
        # 'stage': 'generate',  # 'generate' or 'solve'
        'stage': 'solve',  # 'generate' or 'solve'

    })
    import time

    start = time.time()
    x = solver.solve(Q.to('cuda'))
    print("Time taken:", time.time() - start)
    print((x == 1).sum(), (x == 0).sum())
    print(x.shape)
