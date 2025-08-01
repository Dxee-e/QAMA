from QAMA.QAMultiheadAttention import QAMultiheadAttention
import torch

ARGS_MODEL = {
    # QUBO model
    "quadratic_coeffecient": 1.00,
    "linear_fix_coeffecient": 0.16,
    "penalty_multi_head_fix_coeffecient": 0.8,
}

ARGS_SOLVER = {
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
        "user_id": None,
        "sdk_code": None,
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
        "sample_num": 50,
        "seed": 0,
        "device": "cuda",
    },
}

batch_size = 128
seq_len = 49
d_model = 256
heads = 8
input_Q = torch.randn(batch_size, seq_len, d_model).to('cuda')  # [batch_size, seq_length, d_model]

model = QAMultiheadAttention(
    d_model=d_model,
    embed_dim=d_model,
    num_heads=heads,
    args_model=ARGS_MODEL,
    args_solver=ARGS_SOLVER,
    enable_solvers='gurobi',
).to('cuda')

x, J, h, H, Hd = model(input_Q, is_return_xJhH=True, solver_name='gurobi')
x = x.detach().cpu().numpy()
x0 = (x==0).sum(axis=(-1,-2))
x1 = (x==1).sum(axis=(-1,-2))
print(f"x0 count, min: {x0.min()}, max: {x0.max()}, mean: {x0.mean()}")
print(f"x1 count, min: {x1.min()}, max: {x1.max()}, mean: {x1.mean()}")