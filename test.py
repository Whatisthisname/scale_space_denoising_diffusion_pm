
# %%
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



# %%
# make a transform to normalize the data and change dtype to float tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
