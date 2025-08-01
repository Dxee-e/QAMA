BATCH_SIZE = 512
NUM_HEADS = 8
SEQ_LEN = 49

from QAMA.QAMultiheadAttention import QAMultiheadAttention

import torch
import math
from tabulate import tabulate
from torch.nn import functional as F
import numpy as np
torch.manual_seed(0)

SOLVER_ARGS = {
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
        "batch_num_process": 16,
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
        "batch_num_process": 1,
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
        "iterations": 1,
        "seed": 0,
        "device": "cuda",
    },
    'hybird': {}
}

MODEL_ARGS = {
    # QUBO model
    "quadratic_coeffecient": 1.0,
    "linear_fix_coeffecient": 0.16,
    "penalty_multi_head_fix_coeffecient": 0.8,
}

model = QAMultiheadAttention(
    d_model=256,
    embed_dim=256,
    num_heads=NUM_HEADS,
    args_model=MODEL_ARGS,
    args_solver=SOLVER_ARGS,
    enable_solvers='all',
).to('cuda')

Q = torch.randn(BATCH_SIZE, SEQ_LEN, 256, device='cuda', dtype=torch.float32)

def calculate_one_solver(name):
    x, J, h, E, Hd = model(Q, solver_name=name, is_return_xJhH=True)
    E = E.sum(dim=(-1, -2))
    np.save(f'./{name}_energy.npy', E.detach().cpu().numpy())
    
# calculate_one_solver('gurobi')
# calculate_one_solver('kaiwu_sa')
# calculate_one_solver('montecarlo_greedy')
# calculate_one_solver('c_sa')
calculate_one_solver('hybird')

