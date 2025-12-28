import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.metrics.lpips import LPIPS
import random

from model.base.blocks import Encoder # use encoder arch as discriminator

def l1(x, y):
    return torch.abs(x - y)
    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.total_steps = config.training.main.max_steps
        
        self.perceptual_weight = config.tokenizer.losses.perceptual_weight
        self.perceptual_subsample = config.tokenizer.losses.perceptual_subsample
        if self.perceptual_weight > 0.0:
            self.perceptual_model = LPIPS().eval()
            for param in self.perceptual_model.parameters():
                param.requires_grad = False

        conf_d = config.discriminator
        self.use_disc = conf_d.use_disc
        self.disc_start = conf_d.disc_start
        self.disc_weight = config.tokenizer.losses.disc_weight
        if self.use_disc:
            self.gp_weight = conf_d.losses.gp_weight
            self.gp_noise = conf_d.losses.gp_noise
            self.centering_weight = conf_d.losses.centering_weight
            
            self.disc_model = Encoder(
                model_size=conf_d.model.model_size,
                patch_size=conf_d.model.patch_size,
                in_channels=3,
                out_channels=1,
                in_grid=config.dataset.in_grid,
                out_tokens=1,
            )


    def forward(self, target, recon, global_step, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon)
        else:
            return self._forward_generator(target, recon, global_step)
        

    def _forward_generator(self, target, recon, global_step):
        loss_dict = {}
        target = target.contiguous()
        recon = recon.contiguous()
        B, C, T, H, W = target.shape

        recon_loss = l1(target, recon).mean()
        loss_dict['recon_loss'] = recon_loss

        perceptual_loss = 0.0
        if self.perceptual_weight > 0.0:
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

        g_loss = 0.0
        if self.use_disc and global_step > self.disc_start:
            target = target.detach().contiguous()

            ############################
            for param in self.disc_model.parameters():
                param.requires_grad = False
            ############################

            logits_real = self.disc_model(target).view(B, -1).mean(-1) # [B], score per sample
            logits_fake = self.disc_model(recon).view(B, -1).mean(-1)
            logits_relative = logits_fake - logits_real
            g_loss = F.softplus(-logits_relative).mean()

            loss_dict['g_loss'] = g_loss
            loss_dict['logits_relative'] = logits_relative # this should fluctuate around 0

        total_loss = (
            recon_loss
            + (self.perceptual_weight * perceptual_loss)
            + (self.disc_weight * g_loss)
        ).mean()

        loss_dict['total_loss'] = total_loss
        return total_loss, {'tokenizer/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}
    

    def _forward_discriminator(self, target, recon):
        loss_dict = {}
        target = target.detach().requires_grad_(True).contiguous()
        recon = recon.detach().requires_grad_(True).contiguous()
        B, C, T, H, W = target.shape

        ############################
        for param in self.disc_model.parameters():
            param.requires_grad = True
        ############################

        logits_real = self.disc_model(target).view(B, -1).mean(-1)
        logits_fake = self.disc_model(recon).view(B, -1).mean(-1)
        logits_relative = logits_real - logits_fake # opposite of g_loss
        d_loss = F.softplus(-logits_relative).mean()
        loss_dict['d_loss'] = d_loss

        # https://www.arxiv.org/pdf/2509.24935
        gradient_penalty = 0.0
        if self.gp_weight > 0.0:
            noise = torch.randn_like(target) * self.gp_noise
            logits_real_noised = self.disc_model(target+noise).view(B, -1).mean(-1)
            logits_fake_noised = self.disc_model(recon+noise).view(B, -1).mean(-1)
            r1_penalty = (logits_real - logits_real_noised)**2
            r2_penalty = (logits_fake - logits_fake_noised)**2

            loss_dict['r1_penalty'] = r1_penalty
            loss_dict['r2_penalty'] = r2_penalty
            gradient_penalty = r1_penalty + r2_penalty

        centering_loss = 0.0
        if self.centering_weight > 0.0:
            centering_loss = ((logits_real + logits_fake) ** 2) / 2
            loss_dict['centering_loss'] = centering_loss

        total_loss = (
            d_loss
            + (self.gp_weight / self.gp_noise**2 * gradient_penalty)
            + (self.centering_weight * centering_loss)
        ).mean()
        
        loss_dict['total_loss'] = total_loss
        return total_loss, {'discriminator/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}