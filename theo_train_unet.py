import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
import argparse

from theo_unet import UNet
from train_mnist import create_mnist_dataloaders
import tqdm

import glob
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument(
        "--unet_levels",
        type=int,
        help="Amount of up/downsample stages in UNET.",
        default=3,
    )
    parser.add_argument(
        "--timesteps", type=int, help="sampling steps of DDPM", default=1000
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=1,
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="set to normal sampling method without clip x_0 which could yield unstable samples",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu training", default=False
    )
    parser.add_argument("--run_name", type=str, help="define run name", required=True)
    parser.add_argument("--img_size", type=int, help="size of image", default="28")
    parser.add_argument("--early_stop", type=int, help="early stop", default=10)
    
    # add flag to toggle loading latest checkpoint automatically
    parser.add_argument("--ckpt", action="store_true", help="load latest checkpoint")



    args = parser.parse_args()

    return args


def main(args):

    print("Training unet for {} epochs".format(args.epochs))
    print("Saving images to {}".format(args.run_name))
    print("image size: {}".format(args.img_size))
    
    model = UNet(
        stages=args.unet_levels, context_size=1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.img_size
    )
    device = "cpu" if args.cpu else "cuda"
    model.to(device)


    loaded_epoch = 0
    if args.ckpt:
        # load latest checkpoint from checkpoint folder
        # find the latest checkpoint in the checkpoints/run_name folder
        
        checkpoints = glob.glob("checkpoints/{}/*.pth".format(args.run_name))
        if len(checkpoints) == 0:
            print("No checkpoints found, starting from scratch")
        checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
        latest_checkpoint = checkpoints[-1]
        loaded_epoch = int(re.findall(r"\d+", latest_checkpoint)[-1])
        print("Loading checkpoint: {}".format(latest_checkpoint))
        model.load_state_dict(torch.load(latest_checkpoint))

    else:
        print("No checkpoint loaded, starting from scratch")
        # remove existing run name directory with checkpoints and images
        import shutil
        shutil.rmtree("images/{}".format(args.run_name), ignore_errors=True)
        shutil.rmtree("checkpoints/{}".format(args.run_name), ignore_errors=True)

    # create run folder:
    os.makedirs("images/{}".format(args.run_name), exist_ok=True)
    os.makedirs("checkpoints/{}".format(args.run_name), exist_ok=True)

    for epoch in range(loaded_epoch, loaded_epoch + args.epochs):
        for i, (images, labels) in enumerate(tqdm.tqdm(train_dataloader)):
            if i > args.early_stop:
                break

            images = images.to(device)

            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, -images)
            loss.backward()
            optimizer.step()

            if i % args.log_freq == 0:
                print(
                    "Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, loss.item())
                )

        # after epoch, save model checkpoint:
        torch.save(
            model.state_dict(),
            "checkpoints/{}/small_model_{}.pth".format(args.run_name, epoch),
        )

        

        # after each epoch, sample one batch of images
        with torch.no_grad():
            (images, labels) = next(iter(test_dataloader))
            images = images.to(device)
            prediction = model(images)
            # save the images to the run_name path
            # stack the images and the predictions together and save them in one image

            torchvision.utils.save_image(
                torch.cat((images, prediction), dim=0),
                "images/{}/small_model_{}.png".format(args.run_name, epoch),
                nrow=8,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
