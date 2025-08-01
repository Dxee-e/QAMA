import torch
from torch import nn
from einops import rearrange
from QAMA.QAMultiheadAttention import QAMultiheadAttention

class QAMA_C_SA(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.attn = QAMultiheadAttention(
            d_model=dim,
            embed_dim=dim_head,
            num_heads=heads,
            enable_solvers='c_sa'
        )

    def forward(self, x, additional_args=None):
        x = self.attn(x, solver_name='c_sa')
        return x

if __name__ == "__main__":
    # Example usage
    attention = QAMA_C_SA(dim=64, heads=8, dim_head=8)
    x = torch.randn(128, 49, 64)  # Batch of 10 sequences of length 20 with dimension 128
    output = attention(x)
    print(output.shape)  # Should be (10, 20, 128)