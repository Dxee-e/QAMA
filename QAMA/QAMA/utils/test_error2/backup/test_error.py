BATCH_SIZE = 128
NUM_HEADS = 8
SEQ_LENGTH = 49
solvers_name = [
    'montecarlo_greedy',
    'gurobi',
    'kaiwu_sa',
]
ARGS_MODEL = {
    # QUBO model
    "quadratic_coeffecient": 1.0,
    "linear_fix_coeffecient": 0.15,
    "penalty_multi_head_fix_coeffecient": 0.55,
}
from QAMA.backend_montecarlo_greedy.solver import Solver as Solver_montecarlo_greedy
from QAMA.backend_gurobi.solver import Solver as Solver_gurobi
from QAMA.backend_kaiwu_sa.solver import Solver as Solver_kaiwu_sa

import torch
import math
from tabulate import tabulate
from torch.nn import functional as F
import numpy as np


J = torch.rand(
    (BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, SEQ_LENGTH),
    dtype=torch.float32,
    device='cuda',
)
h = torch.rand(
    (BATCH_SIZE, NUM_HEADS, SEQ_LENGTH),
    dtype=torch.float32,
    device='cuda',
)
J = (J + J.transpose(-1, -2)) / 2  # symmetric

def norm3d(x):
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True)
    return (x - mean) / (std + 1e-12)
def norm4d(x):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True)
    return (x - mean) / (std + 1e-12)
h = norm3d(h)
J = norm4d(J)

gurobi_solver = Solver_gurobi(args_model=ARGS_MODEL, args_solver={'batch_num_process': 16})
kaiwu_sa_solver = Solver_kaiwu_sa(args_solver={'user_id': '69878024601862146', 'sdk_code': '0i4T6LY1XygfwN3MWa8Fjq27OaT0sq', 'batch_num_process': 16}, args_model=ARGS_MODEL)
montecarlo_greedy_solver = Solver_montecarlo_greedy(args_model=ARGS_MODEL, args_solver={'sample_num': 200})

gurobi_x = gurobi_solver.solve(J, h)
kaiwu_sa_x = kaiwu_sa_solver.solve(J, h)
montecarlo_greedy_x = montecarlo_greedy_solver.solve(J, h)

def calculate_term(J, h, x):
    x = torch.from_numpy(x).to(J.device)
    quadratic_term = torch.sum(J * x[:, :, :, None] * x[:, :, None, :], dim=(1,2,3))
    liner_term = torch.sum(h * x, dim=(1,2))
    
    penalty_term = []
    for i in range(NUM_HEADS):
        for j in range(i + 1, NUM_HEADS):
            temp = torch.sum(x[:, i, :] * x[:, j, :], dim=1)
            penalty_term.append(temp)
    penalty_term = torch.stack(penalty_term, dim=1)
    penalty_term = torch.sum(penalty_term, dim=1)
    
    quadratic_term = quadratic_term * ARGS_MODEL["quadratic_coeffecient"]
    liner_term = liner_term * ARGS_MODEL["linear_fix_coeffecient"] * SEQ_LENGTH
    penalty_term = penalty_term * ARGS_MODEL["penalty_multi_head_fix_coeffecient"] * (math.sqrt(2 / math.pi) * SEQ_LENGTH / (NUM_HEADS - 1))
    
    return -quadratic_term, -liner_term, penalty_term

gurobi_quadratic_term, gurobi_liner_term, gurobi_penalty_term = calculate_term(J, h, gurobi_x)
kaiwu_sa_quadratic_term, kaiwu_sa_liner_term, kaiwu_sa_penalty_term = calculate_term(J, h, kaiwu_sa_x)
montecarlo_greedy_quadratic_term, montecarlo_greedy_liner_term, montecarlo_greedy_penalty_term = calculate_term(J, h, montecarlo_greedy_x)

np.savez('./test_error.npz',
         gurobi_quadratic_term=gurobi_quadratic_term.cpu().numpy(),
         gurobi_liner_term=gurobi_liner_term.cpu().numpy(),
         gurobi_penalty_term=gurobi_penalty_term.cpu().numpy(),
         kaiwu_sa_quadratic_term=kaiwu_sa_quadratic_term.cpu().numpy(),
         kaiwu_sa_liner_term=kaiwu_sa_liner_term.cpu().numpy(),
         kaiwu_sa_penalty_term=kaiwu_sa_penalty_term.cpu().numpy(),
         montecarlo_greedy_quadratic_term=montecarlo_greedy_quadratic_term.cpu().numpy(),
         montecarlo_greedy_liner_term=montecarlo_greedy_liner_term.cpu().numpy(),
         montecarlo_greedy_penalty_term=montecarlo_greedy_penalty_term.cpu().numpy()
)