import os
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_checkpoint(model, run_name) -> str:
    checkpoint_folder = "checkpoints/" + run_name
    # get all checkpoints and choose the latest:
    checkpoints = os.listdir(checkpoint_folder)
    checkpoints = [c for c in checkpoints if c.endswith(".pth")]
    checkpoints.sort(key = lambda x: int(x.split("_")[1].split(".")[0]))
    latest = checkpoints[-1]

    # load the checkpoint:
    data = torch.load(os.path.join(checkpoint_folder, latest))
    model.load_state_dict(data)

    return model


def create_mnist_dataloaders(batch_size,image_size=28,num_workers=0):
    
    def map(x):
        newRange =(-3, 3)
        width = newRange[1] - newRange[0]
        return width*x-width/2.0

    preprocess=transforms.Compose([transforms.Resize(image_size, antialias=True),\
                                    transforms.ToTensor(),
                                    # rescale the images from [0, 1] to [-1, 1] range with a linear transformation
                                    transforms.Normalize((0.0), (1.0))])
                                    # transforms.Lambda(map)])

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )

    test_dataset=MNIST(root="./mnist_data",\
                train=False,\
                download=True,\
                transform=preprocess
                )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader
