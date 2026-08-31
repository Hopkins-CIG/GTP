import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math
from .edm_unet import DhariwalUNet
from .karras_diffusion import KarrasDenoiser

class CMSI(pl.LightningModule):
    def __init__(self,
                    img_resolution=256,
                    img_channels=1,
                    model_channels=64,
                    sigma_min=0.0001,
                    sigma_max=1,
                    sigma_data=0.5,
                    cov=0.125,
                    channel_mult=[1, 2, 3, 4, 8],
                    attn_resolutions=[16, 8],
                    use_source=True,
                    use_endpoint=True,
                    conditioning_channels=1,
                    conditioning_resolution=256,
                 ):
        
        super().__init__()
        
        self.unet = DhariwalUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            conditioning_channels=conditioning_channels,
            conditioning_resolution=conditioning_resolution,
        )

        self.denoiser = KarrasDenoiser(sigma_data,
                                       sigma_max,
                                       sigma_min,
                                       cov_xy=cov,
                                       image_size=img_resolution,
                                       weight_schedule='bridge_karras',
                                       pred_mode='vp',
                                       loss_norm=None)
        
        self.img_resolution = img_resolution

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.cov = cov

        self.use_source = use_source
        self.use_endpoint = use_endpoint

        self.save_hyperparameters()

    # Note that in this implementation t is sigma
    def forward(self, x, y, t):
        _, denoised = self.denoiser.denoise(self.unet, x, y, t)
        return denoised
    
    def sample_t(self, n):
        return torch.rand(n) * (self.sigma_max - self.sigma_min) + self.sigma_min
    
    def training_step(self, batch, batch_idx):
        x0 = batch['target']

        if self.use_source and self.use_endpoint: # DDBM
            xT = batch['source']
            y = xT
        if self.use_source and not self.use_endpoint: # Conditional diffusion
            xT = torch.randn_like(x0)
            y = batch['source']
        if not self.use_source: # Unconditional diffusion (but still condition on the endpoint)
            xT = torch.randn_like(x0)
            y = xT

        if batch.get('ct_enc') is not None:
            ct_enc = batch['ct_enc']
            y = ct_enc
            xT = batch['ct_endpoint']
            if xT.shape[-1] != self.img_resolution:
                xT = F.interpolate(xT, size=(self.img_resolution, self.img_resolution), mode='bilinear', align_corners=False)

        t = self.sample_t(x0.shape[0]).to(x0.device)
        
        loss = self.denoiser.training_bridge_losses(self.unet, x0, xT, y, t)

        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x0 = batch['target']
        
        if self.use_source and self.use_endpoint: # DDBM
            xT = batch['source']
            y = xT
        if self.use_source and not self.use_endpoint: # Conditional diffusion
            xT = torch.randn_like(x0)
            y = batch['source']
        if not self.use_source: # Unconditional diffusion (but still condition on the endpoint)
            xT = torch.randn_like(x0)
            y = xT

        if batch.get('ct_enc') is not None:
            ct_enc = batch['ct_enc']
            y = ct_enc
            xT = batch['ct_endpoint']
            if xT.shape[-1] != self.img_resolution:
                xT = F.interpolate(xT, size=(self.img_resolution, self.img_resolution), mode='bilinear', align_corners=False)
                
        t = self.sample_t(x0.shape[0]).to(x0.device)

        loss = self.denoiser.training_bridge_losses(self.unet, x0, xT, y, t)

        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

    
    def configure_callbacks(self):
        val_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='best-val',
            mode='min'
            )
        train_callback = pl.callbacks.ModelCheckpoint(
            monitor='train_loss',
            filename='best-train',
            mode='min',
        )
        save_every_n_steps = 10_000
        save_callback = pl.callbacks.ModelCheckpoint(
            every_n_train_steps=save_every_n_steps,
            filename='step-{step}',
        )
        last_callback = pl.callbacks.ModelCheckpoint(
            filename='last',
        )
        return [val_callback, train_callback, save_callback, last_callback]
    


