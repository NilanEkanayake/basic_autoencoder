import torch
import torch.nn as nn
from model.base.blocks import Encoder, Decoder
from model.quantizer.fsq import FSQ

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        titok_conf = config.tokenizer.model
        in_grid = config.dataset.in_grid
        token_size = len(titok_conf.fsq_levels)

        self.encoder = Encoder(
            model_size=titok_conf.encoder_size,
            patch_size=titok_conf.patch_size,
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=titok_conf.num_tokens,
        )
        self.quantize = FSQ(levels=titok_conf.fsq_levels)
        self.decoder = Decoder(
            model_size=titok_conf.decoder_size,
            patch_size=titok_conf.patch_size,
            in_channels=token_size,
            out_channels=3,
            in_tokens=titok_conf.num_tokens,
            out_grid=in_grid,
        )

    def encode(self, x):
        x = self.encoder(x)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def decode_indices(self, indices):
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype)
        return self.decoder(x_q)
    
    def forward(self, x):
        x_q, out_dict = self.encode(x)
        x = self.decode(x_q)
        return x, out_dict