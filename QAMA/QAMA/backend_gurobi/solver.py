import numpy as np
import torch
import torch.multiprocessing as mp
from icecream import ic
from torch import Tensor
from gurobipy import quicksum
import gurobipy

class Solver:
    def __init__(self, args_solver: dict):
        self.args_solver = args_solver
        
    def _solve_step(self, Q: np.ndarray, env=None) -> np.ndarray:
        # no batch for parallel

        if env is None:
            env = gurobipy.Env()
            env.setParam("OutputFlag", 0)
            env.setParam("LogToConsole", 0)
            env.setParam("SolutionLimit", 1)
            env.setParam("MIPGap", self.args_solver["mipgap"])
            env.setParam("Threads", self.args_solver["gurobi_num_threads"])
            env.start()

        m = Q.shape[0]

        model = gurobipy.Model(env=env)
        x = model.addVars(m, vtype=gurobipy.GRB.BINARY)
        model.update()
        objective = quicksum(
            quicksum(
                Q[i, j] * x[i] * x[j] for j in range(m)
            ) for i in range(m)
        )

        # solve
        model.setObjective(objective, gurobipy.GRB.MINIMIZE)
        model.optimize()
        x_val = np.array([x[i].X for i in range(m)])
        del model
        return x_val

    def _warpper_parallel_solve_step(self, Q: np.ndarray) -> np.ndarray:
        batch = Q.shape[0]

        env = gurobipy.Env()
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.setParam("SolutionLimit", 1)
        env.setParam("MIPGap", self.args_solver["mipgap"])
        env.setParam("Threads", self.args_solver["gurobi_num_threads"])
        env.start()

        results = []
        for b in range(batch):
            Q_b = Q[b, :, :]
            x = self._solve_step(Q_b, env)
            results.append(x)
        results = np.stack(results, axis=0)
        del env
        return results

    def solve(self, Q: Tensor) -> Tensor:
        Q_device = Q.device
        dtype = Q.dtype
        Q = Q.detach().cpu().numpy()
        batch = Q.shape[0]

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
        return torch.from_numpy(results).type(dtype).to(Q_device)


if __name__ == "__main__":
    device = 'cuda'
    torch.manual_seed(0)
    batch_size = 128
    qubit_num = 8*49
    Q = torch.randn(batch_size, qubit_num, qubit_num, device=device, dtype=torch.float32)
    Q = (Q + Q.transpose(-1, -2)) / 2 
    solver = Solver({
        # parallel
        "batch_num_process": 16,
        "timeout": 10,
        # gurobi solver parameters
        "mipgap": 0.01,
        "gurobi_num_threads": 1,
    })
    import time

    start = time.time()
    x = solver.solve(Q)
    print("Time taken:", time.time() - start)
    print((x == 1).sum(), (x == 0).sum())
    print(x.shape)

