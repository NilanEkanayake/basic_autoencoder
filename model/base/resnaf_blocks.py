import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from model.base.utils import get_model_dims, init_weights
import math

class PEG3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            groups=dim
        )
    
    def forward(self, x):
        x = rearrange(x, 'b t h w c -> b c t h w')
        x = self.ds_conv(x.contiguous())
        x = rearrange(x, 'b c t h w -> b t h w c')
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def ffd(dim, mult=4, dropout=0.):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class ResNAF(nn.Module):
    def __init__(self,
                 num_layer, 
                 dim,
                 mlp_ratio=4,
                 ): 
        super(ResNAF, self).__init__()
        self.num_layer = num_layer
        self.dconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, mlp_ratio)) 
            self.dconv_layer.append(PEG3D(dim))
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = x + self.dconv_layer[i](x)
            x = x + self.ffd_layer[i](x) 
        return x
    

class Encoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3, # RGB
            out_channels=5, # len(fsq_levels)
            in_grid=(32, 256, 256),
            **kwargs,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels
        self.grid = [x//y for x, y in zip(in_grid, patch_size)]
        self.width, self.num_layers, _, mlp_ratio = get_model_dims(model_size) # no heads
        
        self.proj_in = nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width)

        self.model_layers = ResNAF(
            dim=self.width,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        x = rearrange(
            x, 'b c (t pt) (h ph) (w pw) -> b t h w (pt ph pw c)',
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]
        )

        x = self.proj_in(x)
        x = self.model_layers(x)
        x = self.proj_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            out_grid=(32, 256, 256),
            **kwargs,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size =in_channels
        self.in_channels = out_channels
        self.grid = [x//y for x, y in zip(out_grid, patch_size)]
        self.width, self.num_layers, _, mlp_ratio = get_model_dims(model_size)

        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)

        self.model_layers = ResNAF(
            dim=self.width,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))
        self.apply(init_weights)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.model_layers(x)
        x = self.proj_out(x)
        
        x = rearrange(
            x, 'b t h w (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
        )
        return x