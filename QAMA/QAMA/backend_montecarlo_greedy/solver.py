import numpy as np
import torch
from torch import Tensor, nn

class Solver:
    def __init__(self, args_solver: dict):
        self.args_solver = args_solver

        self.generator = torch.Generator(device=self.args_solver["device"])
        self.generator.manual_seed(self.args_solver["seed"])

    @torch.no_grad
    def solve_step(self, Q: Tensor) -> Tensor:
        # Q: [b, t*n, t*n]
        device = Q.device

        b, qubit_num, _ = Q.shape
        sample_num = self.args_solver["sample_num"]

        best_x = None
        best_energy = None
        for cur_iter in range(self.args_solver["iterations"]):
            sample_x = torch.randint(
                0,
                2,
                (b, sample_num, qubit_num),
                dtype=torch.float32,
                generator=self.generator,
                device=device,
            )
            min_energy = torch.einsum("bsq,bqr,bsr->bs", sample_x, Q, sample_x)

            for i in range(qubit_num):
                reversed_sample_x = sample_x.clone()
                reversed_sample_x[:, :, i] = 1.0 - sample_x[:, :, i]

                energy = torch.einsum(
                    "bsq,bqr,bsr->bs", reversed_sample_x, Q, reversed_sample_x
                )

                mask = energy < min_energy
                min_energy = torch.where(mask, energy, min_energy)

                sample_x[:, :, i] = torch.where(
                    mask, reversed_sample_x[:, :, i], sample_x[:, :, i]
                )

            # min_idx = min_energy.argmin(dim=1)
            min_energy_values, min_idx = min_energy.min(dim=1)
            best_sample = sample_x[torch.arange(b), min_idx, :]
            
            if cur_iter == 0:
                best_x = best_sample
                best_energy = min_energy_values
            else:
                better_mask = min_energy_values < best_energy
                best_x = torch.where(
                    better_mask.unsqueeze(1), best_sample, best_x
                )
                best_energy = torch.where(better_mask, min_energy_values, best_energy)
                
        return best_x

    @torch.no_grad
    def solve(self, Q: Tensor) -> Tensor:
        device = self.args_solver["device"]
        Q_device = Q.device
        if Q.device != device:
            Q = Q.to(device)

        b, m, _ = Q.shape
        results = self.solve_step(Q)
        return results.to(Q_device)


if __name__ == "__main__":
    device = 'cuda'
    torch.manual_seed(0)
    batch_size = 128
    qubit_num = 8*49
    Q = torch.randn(batch_size, qubit_num, qubit_num, device=device, dtype=torch.float32)
    Q = (Q + Q.transpose(-1, -2)) / 2 
    solver = Solver({
        "sample_num": 50,
        "seed": 0,
        "device": "cuda",
    })
    import time

    start = time.time()
    x = solver.solve(Q)
    print("Time taken:", time.time() - start)
    print((x == 1).sum(), (x == 0).sum())
    print(x.shape)
