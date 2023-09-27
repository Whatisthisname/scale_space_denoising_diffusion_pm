import math
from theo_unet import UNet

import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, image_size, time_embedding_dim=256, timesteps=1000, stages = 3):
        super().__init__()
        self.timesteps = timesteps
        self.image_size = image_size
        self.model = UNet(stages, time_embedding_dim)
        
        self.register_buffer("betas", self._cosine_variance_schedule(timesteps))
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alphas_cumprod", self.alphas.cumprod(dim=-1))
        self.register_buffer("sqrt_alphas_cumprod", self.alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1. - self.alphas_cumprod).sqrt())

    def train(self, clean_image : torch.Tensor):
        """Train the model on a batch of clean images, letting the model predict the noise and returning the MSE. Minimize the output directly."""
        noise = torch.randn_like(clean_image)
        t = torch.randint(0, self.timesteps, (clean_image.shape[0],)).to(clean_image.device)
        #q(x_{t}|x_{0})
        noisy = self.sqrt_alphas_cumprod.gather(-1,t).reshape(clean_image.shape[0],1,1,1)*clean_image+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(clean_image.shape[0],1,1,1)*noise
        
        pred_noise = self.model(noisy, t)
        return torch.mean((pred_noise - noise)**2)
    


def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
    steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
    f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
    betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

    return betas