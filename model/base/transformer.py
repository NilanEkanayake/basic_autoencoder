import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func
from einops import rearrange
import math


class GEGLU(nn.Module):
    def __init__(self):
        super(GEGLU, self).__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return  F.gelu(gate) * x
    
    
def ffd(dim, mult=4, mult_of=32):
    inner_dim = int(mult * (2 / 3) * dim)
    inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim//self.heads

        self.to_qkv = nn.Linear(dim, dim*4, bias=False) # x4 = q, k, v, gate
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v, gate = self.to_qkv(x).chunk(4, dim=-1)

        q = q.unflatten(-1, (self.heads, self.head_dim))
        k = k.unflatten(-1, (self.heads, self.head_dim))
        v = v.unflatten(-1, (self.heads, self.head_dim))

        q = self.q_norm(q.contiguous()).to(q)
        k = self.k_norm(k.contiguous()).to(k)
        x = flash_attn_func(q, k, v)

        x = x.flatten(-2).contiguous()
        x = x * torch.sigmoid(gate) # gating from qwen3-next
        return self.out_proj(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            heads=8,
            mlp_ratio=4,
            num_layer=2,
        ): 
        super(ResidualAttentionBlock, self).__init__()
        self.num_layer = num_layer
        self.attn_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.attn_layer.append(Attn(embed_dim, heads))
            self.ffd_layer.append(ffd(embed_dim, mlp_ratio)) 
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = x + self.attn_layer[i](x.contiguous())
            x = x + self.ffd_layer[i](x.contiguous())

            # LNS (https://arxiv.org/abs/2502.05795) - see section on ViT
            x = x * (1 / math.sqrt(i + 1))
        return x