import torch
import torch.nn as nn


# create MNIST classifier with convolutional layers and 10 outputs for use with softmax.
class MNISTCLF(nn.Module):
    def __init__(self, img_size : int) -> None:
        super().__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (self.img_size // 4 - 1) ** 2, 128)
        self.fc2 = nn.Linear(128, 10)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.gelu(self.conv1(x)))
        x = self.pool(self.gelu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    