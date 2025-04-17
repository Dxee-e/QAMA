from torch import nn
from torch import Tensor

class QAMultiheadAttention(nn.Module):
    def __init__(self, num_heads: int, args_model: dict=None, solver_backend: str='gurobi', args_solver: dict=None):
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
        
        self.num_heads = num_heads
        self.solver_backend = solver_backend
        self.solver = None
        if solver_backend == 'kaiwu_sa':
            from .backend_kaiwu_sa.solver import Solver
            self.solver = Solver(args_model=args_model, args_solver=args_solver)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        pass