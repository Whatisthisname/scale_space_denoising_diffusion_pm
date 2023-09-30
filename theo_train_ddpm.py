import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
import argparse

from theo_DDPM import DDPM
from train_mnist import create_mnist_dataloaders
import tqdm

import glob
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--small_epochs", type=int, default=10)
    parser.add_argument("--big_epochs", type=int, default=10)
    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument(
        "--stages",
        type=int,
        help="Amount of up/downsample stages in UNET.",
        default=3,
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=5,
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="set to normal sampling method without clip x_0 which could yield unstable samples",
    )
    parser.add_argument("--run_name", type=str, help="define run name", required=True)
    parser.add_argument("--img_size", type=int, help="size of image", default="28")
    parser.add_argument("--early_stop", type=int, help="early stop", default=10)
    
    # add flag to toggle loading latest checkpoint automatically
    parser.add_argument("--ckpt", action="store_true", help="load latest checkpoint")

    # timesteps argument
    parser.add_argument("--timesteps", type=int, help="sampling steps of DDPM", default=300)

    args = parser.parse_args()

    return args


def main(args):

    print("Training small model for {} epochs".format(args.small_epochs))
    print("Training big model for {} epochs".format(args.big_epochs))
    print("Saving images to {}".format(args.run_name))
    print("image size: {}".format(args.img_size))
    
    small_model = DDPM(
        args.img_size, 1, args.timesteps, args.stages)  

    optimizer = torch.optim.AdamW(small_model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.img_size
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    small_model.to(device)


    small_loaded_epoch = 0
    if args.ckpt:
        # load latest checkpoint from checkpoint folder
        # find the latest checkpoint in the checkpoints/run_name folder
        
        checkpoints = glob.glob("checkpoints/{}/*.pth".format(args.run_name))
        if len(checkpoints) == 0:
            print("No checkpoints found, starting from scratch")
        else:
            checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
            latest_checkpoint = checkpoints[-1]
            small_loaded_epoch = int(re.findall(r"\d+", latest_checkpoint)[-1])
            print("Loading checkpoint: {}".format(latest_checkpoint))
            small_model.load_state_dict(torch.load(latest_checkpoint))

    else:
        print("No checkpoint loaded, starting from scratch")
        # remove existing run name directory with checkpoints and images
        import shutil
        shutil.rmtree("images/{}".format(args.run_name), ignore_errors=True)
        shutil.rmtree("checkpoints/{}".format(args.run_name), ignore_errors=True)

    # create run folder:
    os.makedirs("images/{}".format(args.run_name), exist_ok=True)
    os.makedirs("checkpoints/{}".format(args.run_name), exist_ok=True)

    for epoch in range(small_loaded_epoch, small_loaded_epoch + args.small_epochs):
        
        with tqdm.tqdm(train_dataloader) as loader:
                
            total_loss = 0
            
            for i, (images, labels) in enumerate(loader):
                if i > args.early_stop:
                    break

                images = images.to(device)

                optimizer.zero_grad()
                loss = small_model.train(images)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                loader.set_description('Epoch %i' % (epoch + 1))
                # Description will be displayed on the left
            
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                loader.set_postfix(epoch_avg_loss=total_loss / (i + 1))




            # after epoch, save model checkpoint:
            torch.save(
                small_model.state_dict(),
                "checkpoints/{}/small_model_{}.pth".format(args.run_name, epoch),
            )



            # after each epoch, sample one batch of images
            with torch.no_grad():
                print("sampling")
                # images = small_model.sample(args.n_samples, return_whole_process=False)
                # images = small_model.sample(8, return_whole_process=True)
                # images = small_model.forward_diffusion(next(iter(loader))[0][:8], keep_intermediate=False, target = int(0.5 * (args.timesteps - 1)))
                images = small_model.forward_diffusion(next(iter(loader))[0][:8], keep_intermediate=True,target=None)
                # save the images to the run_name path
                # stack the images and the predictions together and save them in one image

                torchvision.utils.save_image(
                    images,
                    "images/{}/small_model_{}.png".format(args.run_name, epoch),
                    nrow=8,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
