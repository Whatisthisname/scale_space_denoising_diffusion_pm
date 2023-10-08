import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
import argparse
import torch_ema as ema

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
        "--unet_stages",
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
    parser.add_argument("--ema_update_freq", type=int, help="ema update freq", default=10)
    
    # add flag to toggle loading latest checkpoint automatically
    parser.add_argument("--ckpt", action="store_true", help="load latest checkpoint")

    # timesteps argument
    parser.add_argument("--markov_states", type=int, help="sampling steps of DDPM", default=300)
    parser.add_argument("--noise_power", type=float, help="noise power of DDPM", default=1.5)

    args = parser.parse_args()

    return args


def main(args):

    print("Training small model for {} epochs".format(args.small_epochs))
    print("Training big model for {} epochs".format(args.big_epochs))
    print("Saving images to {}".format(args.run_name))
    print("image size: {}".format(args.img_size))
    
    small_model = DDPM(
        args.img_size, ctx_sz=1+10, markov_states=args.markov_states, unet_stages=args.unet_stages, noise_schedule_param=args.noise_power)
    ema_model = ema.ExponentialMovingAverage(small_model.parameters(), decay=0.95)

    optimizer = torch.optim.AdamW(small_model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.img_size
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    small_model.to(device)


    def from_scratch():
        # remove existing run name directory with checkpoints and images
        import shutil
        shutil.rmtree("images/{}".format(args.run_name), ignore_errors=True)
        shutil.rmtree("checkpoints/{}".format(args.run_name), ignore_errors=True)

        # save arguments to run_name folder, create it if it doesn't exist
        os.makedirs("checkpoints/{}".format(args.run_name), exist_ok=True)
        with open("checkpoints/{}/args.txt".format(args.run_name), "w") as f:
            # for each argument, write it to the file, so it can be directly copied to the terminal
            for arg in vars(args):

                f.write("--{} {}\n".format(arg, getattr(args, arg)))


    small_loaded_epoch = 0
    if args.ckpt:
        # load latest checkpoint from checkpoint folder
        # find the latest checkpoint in the checkpoints/run_name folder
        
        checkpoints = glob.glob("checkpoints/{}/*.pth".format(args.run_name))
        if len(checkpoints) == 0:
            print("No checkpoints found, starting from scratch")
            from_scratch()
        else:
            checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
            latest_checkpoint = checkpoints[-1]
            small_loaded_epoch = 1 + int(re.findall(r"\d+", latest_checkpoint)[-1])
            print("Loading checkpoint: {}".format(latest_checkpoint))
            small_model.load_state_dict(torch.load(latest_checkpoint, map_location=device))

    else:
        print("No checkpoint loaded, starting from scratch")
        from_scratch()
        

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
                labels = labels.to(device)
                # print(labels)
                # one-hot encode the digits
                labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
                # print(labels)
                # exit()

                optimizer.zero_grad()
                loss = small_model.train(images, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

                loader.set_description('E%i' % (epoch + 1))
                # Description will be displayed on the left
            
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                loader.set_postfix(avg_loss=total_loss / (i + 1))

                if i % args.ema_update_freq == 0:
                    ema_model.update()




            # after epoch, save model checkpoint:
            torch.save(
                small_model.state_dict(),
                "checkpoints/{}/small_model_{}.pth".format(args.run_name, epoch),
            )



            # after each epoch, sample one batch of images
            # with torch.no_grad():
            with ema_model.average_parameters():
                # break
                print("sampling")
                # target_label = torch.randint(0, 10, (args.n_samples,)).tolist()
                # images = small_model.sample(20, [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9], return_whole_process=True)
        
                sample_data = next(iter(test_dataloader))
                input_images =sample_data[0][:args.n_samples].to(device)
                input_labels = sample_data[1][:args.n_samples].tolist()
    
                forward_images = small_model.forward_diffusion(input_images, None, keep_intermediate=True, target=None)
                reverse_images = small_model.sample(args.n_samples, input_labels, return_whole_process=True)
                
                # print(forward_images.shape)
                # print(reverse_images.shape)


                # # rotate the reverse images by 180 degrees
                # reverse_images = torch.flip(reverse_images, dims=[2,3])

                # concat the forward and reverse images:
                images = torch.cat((forward_images, reverse_images), dim=2)
                # images = forward_images

                

                # save the images to the run_name path
                # stack the images and the predictions together and save them in one image

                torchvision.utils.save_image(
                    images,
                    "images/{}/small_model_{}.png".format(args.run_name, epoch),
                    nrow=20,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
