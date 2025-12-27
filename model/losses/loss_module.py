import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.metrics.lpips import LPIPS
import random


def l1(x, y):
    return torch.abs(x - y)
    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.total_steps = config.training.main.max_steps
        self.perceptual_weight = config.tokenizer.model.perceptual_weight
        self.perceptual_subsample = config.tokenizer.model.perceptual_subsample
        
        if self.perceptual_weight > 0.0:
            self.perceptual_model = LPIPS().eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False
        

    def forward(self, target, recon):
        loss_dict = {}

        target = target.contiguous()
        recon = recon.contiguous()

        recon_loss = l1(target, recon).mean()
        loss_dict['recon_loss'] = recon_loss

        perceptual_loss = 0.0
        if self.perceptual_weight > 0.0:
            B, C, T, H, W = target.shape
            num_sub = self.perceptual_subsample
            if num_sub != -1 and num_sub < B*T:
                sub_idx = torch.randperm(B*T, device=target.device)[:num_sub]
            else:
                sub_idx = torch.arange(B*T, device=target.device)

            perceptual_loss = self.perceptual_model(
                rearrange(recon, 'b c t h w -> (b t) c h w')[sub_idx],
                rearrange(target, 'b c t h w -> (b t) c h w')[sub_idx],
            ).mean()
            loss_dict['perceptual_loss'] = perceptual_loss

        total_loss = (
            recon_loss
            + (self.perceptual_weight * perceptual_loss)
        ).mean()

        loss_dict['total_loss'] = total_loss
        return total_loss, {k:v.clone().mean().detach() for k,v in loss_dict.items()}