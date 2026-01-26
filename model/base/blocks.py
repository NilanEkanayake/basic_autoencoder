import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.transformer import ResidualAttentionBlock
from model.base.utils import get_model_dims, init_weights
from model.base.rope import get_freqs

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
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        self.freqs = get_freqs(out_tokens, self.grid, head_dim=self.width//self.heads)

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
        device = x.device

        x = rearrange(
            x, 'b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)',
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]
        )
        x = self.proj_in(x) # returns BLC

        mask_tokens = self.mask_token.expand(B, self.out_tokens, self.width)
        x = torch.cat([mask_tokens, x], dim=1)

        x = self.model_layers(x, freqs=self.freqs.to(device))

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
        self.grid_size = math.prod(self.grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        self.freqs = get_freqs(in_tokens, self.grid, head_dim=self.width//self.heads)

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
        device = x.device

        x = self.proj_in(x)

        mask_tokens = self.mask_token.expand(B, self.grid_size, self.width)
        x = torch.cat([x, mask_tokens], dim=1)

        x = self.model_layers(x, freqs=self.freqs.to(device))

        x = x[:, self.in_tokens:]
        x = self.proj_out(x)
        x = rearrange(
            x, 'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
            t=self.grid[0], h=self.grid[1], w=self.grid[2],
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
        )

        return x
