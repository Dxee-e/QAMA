import kaiwu as kw
import numpy as np
import torch
import torch.multiprocessing as mp
from icecream import ic
from torch import Tensor

class Solver:
    def __init__(self, args_solver: dict):
        self.args_solver = args_solver
        # init kaiwu SDK
        kw.license.init(self.args_solver["user_id"], self.args_solver["sdk_code"])

    def _solve_step(self, Q: np.ndarray) -> np.ndarray:
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
        worker = kw.solver.SimpleSolver(
            kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=self.args_solver["initial_temperature"],
                alpha=self.args_solver["alpha"],
                cutoff_temperature=self.args_solver["cutoff_temperature"],
                iterations_per_t=self.args_solver["iterations_per_t"],
                size_limit=self.args_solver["size_limit"],
                rand_seed=self.args_solver["rand_seed"],
                process_num=self.args_solver["sa_num_process"],
            )
        )
        sol_dict, _ = worker.solve_qubo(qubo_model)
        x = kw.qubo.get_array_val(x, sol_dict)
        return x

    def _warpper_parallel_solve_step(self, Q: np.ndarray) -> np.ndarray:
        batch = Q.shape[0]
        results = []
        for b in range(batch):
            Q_b = Q[b, :, :]
            x = self._solve_step(Q_b)
            results.append(x)
        results = np.stack(results, axis=0)
        return results

    def solve(self, Q: Tensor) -> Tensor:
        device = Q.device
        dtype = Q.dtype
        batch = Q.shape[0]
        Q = Q.detach().cpu().numpy()
        
        if self.args_solver["batch_num_process"] == 1 or batch == 1:
            results = []
            for b in range(batch):
                results.append(self._solve_step(Q[b, :, :]))
            results = np.stack(results, axis=0)
        else:
            num_process = (
                self.args_solver["batch_num_process"]
                if batch > self.args_solver["batch_num_process"]
                else batch
            )

            chunk_size = batch // num_process
            chunk_extra_size = batch % num_process
            input_list = []
            start_idx = 0
            for i in range(num_process):
                if chunk_extra_size > 0:
                    end_idx = start_idx + chunk_size + 1
                    chunk_extra_size -= 1
                else:
                    end_idx = start_idx + chunk_size
                input_list.append((Q[start_idx:end_idx, :, :], ))
                start_idx = end_idx

            with mp.Pool(num_process) as pool:
                results = pool.starmap_async(
                    self._warpper_parallel_solve_step, input_list
                )
                try:
                    results = results.get(timeout=self.args_solver['timeout']*chunk_size)
                except mp.TimeoutError:
                    ic("TimeoutError")
                    pool.terminate()
                    pool.join()
                    exit(-1)
                except Exception as e:
                    ic("Exception", e)
                    pool.terminate()
                    pool.join()
                    exit(-1)
            results = np.concatenate(results, axis=0)
            
        return torch.from_numpy(results).type(dtype).to(device)
    

if __name__ == "__main__":
    device = 'cuda'
    torch.manual_seed(0)
    batch_size = 128
    qubit_num = 8*49
    Q = torch.randn(batch_size, qubit_num, qubit_num, device=device, dtype=torch.float32)
    Q = (Q + Q.transpose(-1, -2)) / 2 
    solver = Solver({
        # kaiwu SDK init license
        "user_id": "69878024601862146",
        "sdk_code": "0i4T6LY1XygfwN3MWa8Fjq27OaT0sq",
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
    })
    import time

    start = time.time()
    x = solver.solve(Q)
    print("Time taken:", time.time() - start)
    print((x == 1).sum(), (x == 0).sum())
    print(x.shape)
