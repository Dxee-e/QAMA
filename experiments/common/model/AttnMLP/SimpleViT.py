import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    if isinstance(t, list):
        return tuple(t)
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_name: str):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        
        if attn_name == "OriginalAttention":
            from .attn_OriginalAttention import OriginalAttention
            self.attention = OriginalAttention(dim=dim, heads=heads, dim_head=dim_head)
        # elif attn_name == "QAMA_C_SA":
        #     from .attn_QAMA_C_SA import QAMA_C_SA
        #     self.attention = QAMA_C_SA(dim=dim, heads=heads, dim_head=dim_head)
        elif attn_name == "LongFormer":
            from .attn_LongFormer import LongFormer
            self.attention = LongFormer(dim=dim, heads=heads, dim_head=dim_head)
        elif attn_name == "BigBird":
            from .attn_BigBird import BigBird
            self.attention = BigBird(dim=dim, heads=heads, dim_head=dim_head)
        elif attn_name == "Linformer":
            from .attn_Linformer import Linformer
            self.attention = Linformer(dim=dim, heads=heads, dim_head=dim_head)
        elif attn_name == "Performer":
            from .attn_Performer import Attention
            self.attention = Attention(dim=dim, heads=heads, dim_head=dim_head)
        elif attn_name == "Reformer":
            from .attn_Reformer import Attention
            self.attention = Attention(dim=dim, heads=heads, dim_head=dim_head)
        else:
            raise ValueError(f"Unknown attention module: {attn_name}.")

    def forward(self, x, additional_args=None):
        x = self.attention(x, additional_args=additional_args)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_name: str):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, attn_name=attn_name),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x, additional_args=None):
        for attn, ff in self.layers:
            x = attn(x, additional_args) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        attn_name: str,
        channels=3,
        dim_head=64,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, attn_name)
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img, additional_args=None):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x, additional_args=additional_args)
        x = x.mean(dim=1)

        return self.linear_head(x)


if __name__ == "__main__":
    # test_attn_names = ['OriginalAttention']
    test_attn_names = ['Reformer']
    # test_attn_names = ['LongFormer']
    for attn_name in test_attn_names:
        model = SimpleViT(
            image_size=(32, 32),
            patch_size=(4, 4),
            num_classes=10,
            dim=64,
            depth=1,
            heads=8,
            mlp_dim=64,
            attn_name=attn_name,
            channels=3,
            dim_head=8
        )
        x = torch.randn(128, 3, 32, 32)
        output = model(x)
        output.mean().backward()
        print(f"Output shape with {attn_name}: {output.shape}")
        # Should be (128, 10) for batch size 128 and 10 classes
        
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                print(f"{name}: {param.shape}")
        print(f"Total parameters in the model with {attn_name}: {total_params}")