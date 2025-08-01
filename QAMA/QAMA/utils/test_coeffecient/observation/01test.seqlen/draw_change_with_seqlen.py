BATCH_SIZE=64
NUM_HEADS=8
SEQLEN_RANGE=range(1, 70)
D_MODEL = 256

import numpy as np
import torch
from QAMA.QAMultiheadAttention import QAMultiheadAttention
from icecream import ic
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd

ARGS_MODEL = {
    # QUBO model
    "quadratic_coeffecient": 1.0,
    "linear_fix_coeffecient": 1.0,
    "penalty_multi_head_fix_coeffecient": 1.0,
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
model = QAMultiheadAttention(
    d_model=D_MODEL,
    embed_dim=D_MODEL,
    num_heads=NUM_HEADS,
    args_model=ARGS_MODEL,
    args_solver=ARGS_SOLVER,
    enable_solvers='montecarlo_greedy',
).to('cuda')
generator = torch.Generator(device='cuda')
generator.manual_seed(0)

def calculate_term(J, h, x):
    assert J[:, :, torch.eye(J.shape[2], dtype=bool)[:, :]].sum() == 0, "J should be symmetric and diagonal elements should be zero"
    assert (J == J.transpose(-1, -2)).all(), "J must be symmetric"
    b, t, n = x.shape
    
    quadratic_term = torch.sum(J * x[:, :, :, None] * x[:, :, None, :], dim=(1,2,3))
    liner_term = torch.sum(h * x, dim=(1,2))
    
    penalty_term = []
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            if i==j:
                continue
            temp = torch.sum(x[:, i, :] * x[:, j, :], dim=1)
            penalty_term.append(temp)
    penalty_term = torch.stack(penalty_term, dim=1)
    penalty_term = torch.sum(penalty_term, dim=1)
    
    quadratic_term = quadratic_term * ARGS_MODEL["quadratic_coeffecient"]
    liner_term = liner_term * ARGS_MODEL["linear_fix_coeffecient"] * n
    penalty_term = penalty_term * ARGS_MODEL["penalty_multi_head_fix_coeffecient"] / 2 * (np.sqrt(2 / np.pi) * n / (t - 1))
    
    return quadratic_term, liner_term, penalty_term

record = []
for seqlen in tqdm(SEQLEN_RANGE):
    Query = torch.randn((BATCH_SIZE, seqlen, D_MODEL), device='cuda', dtype=torch.float32, generator=generator)
    x, J, h, H, Hd = model(Query, solver_name='montecarlo_greedy', is_return_xJhH=True)
    quadratic_term, liner_term, penalty_term = calculate_term(J, h, x)
    q2l = quadratic_term / liner_term
    q2p = quadratic_term / penalty_term
    
    # q2l = q2l.mean().item()
    # q2p = q2p.mean().item()
    
    record.append([seqlen, q2l, q2p])


# Prepare data for boxplot
data = []
for seqlen, q2l, q2p in record:
    # q2l and q2p are tensors, flatten and pair with seqlen
    q2l = q2l.detach().cpu().numpy().flatten()
    q2p = q2p.detach().cpu().numpy().flatten()
    for val in q2l:
        data.append({'SEQLEN': seqlen, 'Type': 'Q2L', 'Value': val})
    for val in q2p:
        data.append({'SEQLEN': seqlen, 'Type': 'Q2P', 'Value': val})

df = pd.DataFrame(data)

plt.figure(figsize=(14, 6))
sns.boxplot(x='SEQLEN', y='Value', hue='Type', data=df, showfliers=False)
plt.title("Boxplot of Q2L and Q2P vs SEQLEN")
plt.xlabel("SEQLEN")
plt.ylabel("Value")
plt.legend(title='Type')
plt.tight_layout()
plt.savefig("Q2L_Q2P_boxplot_with_seqlen.png")
plt.close()

