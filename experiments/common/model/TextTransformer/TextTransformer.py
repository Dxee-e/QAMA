import torch
from torch import nn
from transformers import AlbertTokenizer, AlbertModel


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
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, solver_name=None):
        batch, seq_len, embed_dim = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = k.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.inner_dim)        
        # return self.to_out(out)
        return out

class TextEmbedder(nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.embed = AlbertModel.from_pretrained('albert-base-v2')
        for param in self.embed.parameters():
            param.requires_grad = False
    
    def forward(self, sequences):
        inputs = self.tokenizer(
            sequences,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {key: value.to(self.embed.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.embed(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x, solver_name=None):
        for attn, ff in self.layers:
            x = attn(x, solver_name=solver_name) + x
            x = ff(x) + x
        return self.norm(x)


class TextTransformer(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_text = TextEmbedder(max_seq_len)
        self.down_embedding = nn.Linear(768, dim, bias=False)
        dim_head = dim // heads
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(dim, num_classes),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, text, solver_name=None):
        x = self.down_embedding(self.embed_text(text))
        x = self.transformer(x, solver_name=solver_name)
        x = x.mean(dim=1)
        return self.cls_head(x)
