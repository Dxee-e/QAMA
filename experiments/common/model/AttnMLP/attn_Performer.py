"""change from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager


APEX_AVAILABLE = False

# helpers

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield



# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# efficient causal linear attention
def causal_linear_attention(q, k, v, eps = 1e-6):
    #k_cumsum = k.cumsum(dim=-2) + eps
    #D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
    #out = causal_dot_product_fn(q, k, v)
    #out = torch.einsum('...nd,...n->...nd', out, D_inv)
    #return out
    
    # q: [b, h, n, d]
    # k: [b, h, n, d]
    # v: [b, h, n, d]
    b, h, n, d = q.shape
    
    # Initialize output and cumulative sums
    out = torch.zeros_like(q)
    k_cumsum = torch.zeros_like(k)
    kv_cumsum = torch.zeros_like(v)
    
    # Iterate through the sequence length
    for i in range(n):
        # Update cumulative sums
        if i > 0:
            k_cumsum[:, :, i] = k_cumsum[:, :, i-1] + k[:, :, i]
            kv_cumsum[:, :, i] = kv_cumsum[:, :, i-1] + torch.einsum('bhdi,bhdi->bhdi', k[:, :, :i+1, :], v[:, :, :i+1, :])
        else:
            k_cumsum[:, :, i] = k[:, :, i]
            kv_cumsum[:, :, i] = torch.einsum('bhdi,bhdi->bhdi', k[:, :, :i+1, :], v[:, :, :i+1, :])
        
        # Calculate D_inv
        D_inv = 1. / (torch.einsum('...nd,...nd->...n', q[:, :, i:i+1], k_cumsum[:, :, i:i+1].type_as(q)) + eps)
        
        # Calculate output
        out[:, :, i] = torch.einsum('...nd,...n->...nd', kv_cumsum[:, :, i], D_inv)
    return out

# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size = 128, eps = 1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim = -2)

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True
    ):
        # small head -> no local head
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        # self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb = None, mask = None, additional_args=None):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = False

        context = x
        # context_mask = None

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out =  self.to_out(out)
        return out


# rotary positional embedding helpers

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k



if __name__ == "__main__":
    # Example usage
    attention = Attention(dim=64, heads=8, dim_head=8)
    x = torch.randn(128, 49, 64)  # Batch of 10 sequences of length 20 with dimension 128
    output = attention(x)
    print(output.shape)  # Should be (10, 20, 128)