import torch
import numpy as np
import kaiwu as kw
from icecream import ic
import torch.multiprocessing as mp

DEFAULT_ARGS_MODEL = {
    # soft selection
    'num_soft_selection_qubit': 4,
    # QUBO model
    'quartic_coeffecient': 1.0,
    'linear_coeffecient': 1.0,
    'penalty_multi_head': 2.0,
}

DEFAULT_ARGS_SOLVER = {
    # kaiwu SDK init license
    'user_id': None,
    'sdk_code': None,
    # parallel
    'batch_num_process': 1,
    'sa_num_process': 1,
    # Simulated Annealing Parameters
    'initial_temperature': 100,
    'alpha': 0.99,
    'cutoff_temperature': 1e-3,
    'iterations_per_t': 10,
    'size_limit': 10,
    'rand_seed': None,
}

class Solver:
    def __init__(self, args_model:dict={}, args_solver:dict={}):
        self.args_model = DEFAULT_ARGS_MODEL.copy()
        self.args_model.update(args_model)
        self.args_solver = DEFAULT_ARGS_SOLVER.copy()
        self.args_solver.update(args_solver)
        
        # init kaiwu SDK
        kw.license.init(self.args_solver['user_id'], self.args_solver['sdk_code'])
        
        n_expand = 2**self.args_model['num_soft_selection_qubit']
        
        # soft selection
        self.soft_bin2dec = np.array([2**i for i in range(self.args_model['num_soft_selection_qubit'])])
        self.soft_selection_value = 1.0 / (n_expand-1)
        
        
    def _solve_step(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        # no batch for parallel
        # J: [t, n, n]
        # h: [t, n]
        t, n, _ = J.shape
        q = self.args_model['num_soft_selection_qubit']
        
        x = kw.qubo.ndarray((t, n, q), "x", kw.qubo.Binary)
        qubo_model = kw.qubo.QuboModel()
        
        xw = np.sum(x*self.soft_bin2dec, axis=-1) * self.soft_selection_value
        
        # objective
        quartic_term = (J * xw[:, :, None] * xw[:, None, :]).sum()
        linear_term = (h * xw).sum()
        objective = -self.args_model['quartic_coeffecient'] * quartic_term - self.args_model['linear_coeffecient'] * linear_term
        qubo_model.set_objective(objective)
        
        # constraints - multi head
        multi_head_term = []
        for i in range(t):
            for j in range(i+1,t):
                multi_head_term.append((x[i, :] * x[j, :]).sum())
        multi_head_term = kw.qubo.quicksum(multi_head_term) / len(multi_head_term)
        qubo_model.add_constraint(multi_head_term, "multi_head", penalty=self.args_model['penalty_multi_head'])
    
        # solve
        worker = kw.solver.SimpleSolver(
            kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=self.args_solver['initial_temperature'],
                alpha=self.args_solver['alpha'],
                cutoff_temperature=self.args_solver['cutoff_temperature'],
                iterations_per_t=self.args_solver['iterations_per_t'],
                size_limit=self.args_solver['size_limit'],
                rand_seed=self.args_solver['rand_seed'],
                process_num=self.args_solver['sa_num_process'],
            )
        )
        sol_dict, _ = worker.solve_qubo(qubo_model)
        x = kw.qubo.get_array_val(x, sol_dict)
        return x    
    
    def _warpper_parallel_solve_step(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        batch = J.shape[0]
        results = []
        for b in range(batch):
            J_b = J[b, :, :, :]
            h_b = h[b, :, :]
            x = self._solve_step(J_b, h_b)
            results.append(x)
        results = np.stack(results, axis=0)
        return results
    
    def solve(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        if self.args_solver['batch_num_process'] == 1:
            batch = J.shape[0]
            results = []
            for b in range(batch):
                results.append(self._solve_step(J[b, :, :, :], h[b, :, :]))
            results = np.stack(results, axis=0)
            return results
        else:
            batch = J.shape[0]
            num_process = self.args_solver['batch_num_process'] if batch > self.args_solver['batch_num_process'] else batch
            
            chunk_size = (batch + num_process - 1) // num_process
            input_list = [(J[i * chunk_size: min((i + 1) * chunk_size, batch), :, :, :], 
                           h[i * chunk_size: min((i + 1) * chunk_size, batch), :, :], 
                          ) for i in range(num_process)]
            
            with mp.Pool(num_process) as pool:
                results = pool.starmap_async(self._warpper_parallel_solve_step, input_list).get()
                pool.close()
                pool.join()
            results = np.concatenate(results, axis=0)
            return results
                    
    
    def energy_function(self, J: torch.Tensor, h: torch.Tensor, x: np.ndarray) -> torch.Tensor:
        # J: [b, t, n, n]
        # h: [b, t, n]
        # x: [b, t, n, q]
        
        xw = np.sum(x*self.soft_bin2dec, axis=-1) * self.soft_selection_value
        xw = torch.from_numpy(xw).type(J.dtype).to(J.device)
        
        quartic_term = torch.sum(J * xw[:, :, :, None] * xw[:, :, None, :], dim=-1)
        linear_term = h * xw
        objective = -self.args_model['quartic_coeffecient'] * quartic_term - self.args_model['linear_coeffecient'] * linear_term
        
        return objective # [batch_size, num_heads]