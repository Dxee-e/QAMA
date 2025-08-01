import torch
from torch import nn
from einops import rearrange

class OriginalAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dim = dim
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, additional_args=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        # return self.to_out(out)
        return out

if __name__ == "__main__":
    # Example usage
    attention = Attention(dim=64, heads=8, dim_head=8)
    x = torch.randn(128, 49, 64)  # Batch of 10 sequences of length 20 with dimension 128
    output = attention(x)
    print(output.shape)  # Should be (10, 20, 128)