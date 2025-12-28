import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.transformer import ResidualAttentionBlock
from model.base.utils import get_model_dims, init_weights

from einops.layers.torch import Rearrange
from einops import rearrange
import math
        
    
class Encoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3, # RGB
            out_channels=5, # len(fsq_levels)
            in_grid=(32, 256, 256),
            out_tokens=2048,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels
        self.out_tokens = out_tokens
        self.grid = [x//y for x, y in zip(in_grid, patch_size)]
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5
        
        self.proj_in = nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width)
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, math.prod(self.grid), self.width))
        self.latent_token_mask = nn.Parameter(scale * torch.randn(1, out_tokens, self.width))

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = rearrange(
            x, 'b c (nt pt) (nh ph) (nw pw) -> b (nt nh nw) (c pt ph pw)',
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]
        )
        x = self.proj_in(x) # returns BLC

        x = x + self.positional_embedding
        latent_token_mask = self.latent_token_mask.expand(B, -1, -1)
        x = torch.cat([latent_token_mask, x], dim=1)

        x = self.model_layers(x)

        x = x[:, :self.out_tokens]
        x = self.proj_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            in_tokens=2048,
            out_grid=(32, 256, 256),
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size =in_channels
        self.in_channels = out_channels
        self.in_tokens = in_tokens
        self.grid = [x//y for x, y in zip(out_grid, patch_size)]
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, in_tokens, self.width))
        self.patch_token_mask = nn.Parameter(scale * torch.randn(1, math.prod(self.grid), self.width)) # T*H*W for total patch token count

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))
        self.apply(init_weights)

    def forward(self, x): # unlike the encoder, 'x' is the quantized latent tokens
        B = x.shape[0]
        x = self.proj_in(x)

        x = x + self.positional_embedding
        patch_token_mask = self.patch_token_mask.expand(B, -1, -1)
        x = torch.cat([x, patch_token_mask], dim=1)

        x = self.model_layers(x)

        x = x[:, self.in_tokens:]
        x = self.proj_out(x)
        x = rearrange(
            x, 'b (nt nh nw) (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)',
            nt=self.grid[0], nh=self.grid[1], nw=self.grid[2],
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
        )

        return x
