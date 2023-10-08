import math
import os
from theo_unet import UNet

import torch
import torchvision
import torch.nn as nn

from train_mnist import create_mnist_dataloaders


class DDPM(nn.Module):
    def __init__(self, image_size, ctx_sz=1, markov_states=1000, unet_stages=3, noise_schedule_param=10.0):
        super().__init__()
        self.markov_states = markov_states
        self.image_size = image_size
        self.model = UNet(unet_stages, ctx_sz)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.register_buffer("betas", _cosine_variance_schedule(markov_states, power=noise_schedule_param))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", self.alphas.cumprod(dim=-1))
        self.register_buffer("sqrt_alphas_cumprod", self.alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - self.alphas_cumprod).sqrt()
        )

    def train(self, clean_image: torch.Tensor, one_hot_labels: torch.Tensor):
        """Train the model on a batch of clean images, letting the model predict the noise and returning the MSE. Minimize the output directly."""
        noise = torch.randn_like(clean_image)
        t = torch.randint(0, self.markov_states-1, (clean_image.shape[0],)).to(
            clean_image.device
        )

        noisy = self.forward_diffusion(clean_image, noise, t, keep_intermediate=False)

        timestep_info = t.unsqueeze(1).float() / (self.markov_states-1)
        context = torch.cat([timestep_info, one_hot_labels], dim=1)
        # print(context.shape)
        # context = timestep_info

        pred_noise = self.model(noisy, context)

        return torch.mean((pred_noise - noise) ** 2)

    def forward_diffusion(
        self, clean_images: torch.Tensor, noise : torch.Tensor, target: torch.Tensor, keep_intermediate: bool
    ) -> torch.Tensor:
        
        
        if keep_intermediate:
            images = [clean_images]

            for t in range(self.markov_states-1):
                image_scale = self.sqrt_alphas_cumprod[t]
                noise_scale = self.sqrt_one_minus_alphas_cumprod[t]
                noised = image_scale * clean_images + noise_scale * torch.randn_like(
                    clean_images
                )
                images.append(noised)

            # concatenate each step into one image for for each sample
            return torch.cat(images, dim=2)

        else:
            image_scale = self.sqrt_alphas_cumprod.gather(0, target).reshape(
                clean_images.shape[0], 1, 1, 1
            )
            noise_scale = self.sqrt_one_minus_alphas_cumprod.gather(0, target).reshape(
                clean_images.shape[0], 1, 1, 1
            )
            return image_scale * clean_images + noise_scale * noise

    @torch.no_grad()
    def sample(self, amount: int, target_label : list[int], return_whole_process: bool) -> torch.Tensor:
        """Sample from the model."""
        # sample noise from standard normal distribution
        image = (
            torch.randn((amount, 1, self.image_size, self.image_size))
            .to(self.device)
            .float()
        )

        tensor_label = torch.tensor(target_label).to(self.device)

        # print("image:", image[0, 0, :, 8])

        images = []
        images.append(image)

        for t in reversed(range(0, self.markov_states-1)):
            t_step = t * torch.ones(amount, dtype=int).to(self.device)
            context = torch.zeros((amount, 1+10)).to(self.device)
            context[:, 0] = t / (self.markov_states-1)
            # context.scatter_(1, tensor_label, value= 1) 
            context[range(amount), tensor_label+1] = 1
            image: torch.Tensor = self.reverse_diffusion(image, t_step, context).clone()
            images.append(image)

        if return_whole_process:
            # images holds the images from the noisiest to the denoised image
            # concatenate each step into one image for for each sample
            images = torch.cat(images, dim=2)
            return images

        else:
            return image

    @torch.no_grad()
    def reverse_diffusion(self, x_t: torch.Tensor, t : torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        """
        # print("x_t:", x_t[0, 0, :, 8])
        # print(t)


        noise_pred = self.model.forward(x_t, context=context)
        # print("pred:", pred[0, 0, :, 8])


        batch_size: int = x_t.shape[0]
        alpha_t = self.alphas.gather(-1, t).reshape(batch_size, 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(batch_size, 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(batch_size, 1, 1, 1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * noise_pred
        )

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                batch_size, 1, 1, 1
            )
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )

        else:
            # print("std = 0")
            std = 0.0

        noise = torch.randn_like(x_t)
        return mean + std * noise


def _cosine_variance_schedule(timesteps, epsilon=0.003, power=10.0):
    steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
    f_t = (
        torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
        ** power
    )
    betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

    return betas



# plot the cosine variance schedule if running this file by itself
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    power = 1.5
    img_size = 32
    n_imgs = 20
    model = DDPM(img_size, markov_states=30, noise_schedule_param = power)



    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=n_imgs, image_size=img_size
    )

    data = next(iter(train_dataloader))
    input_images = data[0][:n_imgs]
    input_labels = data[1][:n_imgs]
    print(input_labels)
    noise = torch.randn_like(input_images)
    noise = torch.randn_like(input_images)

    images = model.forward_diffusion(input_images, noise, keep_intermediate=True,target=None)

    # save the images locally
    # create the images folder if it doesn't exist

    os.makedirs("images/schedules", exist_ok=True)

    torchvision.utils.save_image(
        images,
        "images/schedules/s.png".format("test", 0),
        nrow=n_imgs,
    )
    
    n = 100
    x_axis = range(n)
    for power in [1, 2, 5, 10, 20, 50, 100]:
        plt.plot(x_axis, _cosine_variance_schedule(n, power = power), label=power)
    plt.legend()
    plt.show()
