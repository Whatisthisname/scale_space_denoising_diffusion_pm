import matplotlib
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
import argparse
import torch_ema as ema
import sys
sys.path.insert(0, '../models/')
from models.DDPM import DDPM
import tqdm

from utils import create_mnist_dataloaders


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
        default=12,
    )
    parser.add_argument(
        "--unet_stages",
        type=int,
        help="Amount of up/downsample stages in UNET.",
        default=3,
    )

    parser.add_argument("--run_name", type=str, help="define run name", required=True)
    parser.add_argument("--img_size", type=int, help="size of image", default="28")
    parser.add_argument("--early_stop", type=int, help="early stop", default=1000)
    
    # add flag to toggle loading latest checkpoint automatically
    parser.add_argument("--ckpt", action="store_true", help="load latest checkpoint")

    # timesteps argument
    parser.add_argument("--markov_states", type=int, help="sampling steps of DDPM", default=50)
    parser.add_argument("--noise_power", type=float, help="noise power of DDPM", default=1.5)

    args = parser.parse_args()

    return args


def main(args):

    print("Saving images to {}".format(args.run_name))
    
    small_model = DDPM(
        args.img_size, ctx_sz=1+10, markov_states=args.markov_states, unet_stages=args.unet_stages, noise_schedule_param=args.noise_power)
    ema_model = ema.ExponentialMovingAverage(small_model.parameters(), decay=0.95)

    

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

        # show a few forward noisings:
        data = next(iter(train_dataloader))
        input_images = data[0][:20]

        noise = torch.randn_like(input_images).to(device)

        images = small_model.forward_diffusion(input_images, noise, keep_intermediate=True, target=None)

        # save the images locally
        # create the images folder if it doesn't exist

        os.makedirs("images/schedules", exist_ok=True)

        torchvision.utils.save_image(
            images,
            "checkpoints/{}/forward.png".format(args.run_name),
            nrow=20,
        )


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
        

    optimizer = torch.optim.Adam(small_model.parameters(), lr=args.lr)

    # create run folder:
    os.makedirs("images/{}".format(args.run_name), exist_ok=True)
    os.makedirs("checkpoints/{}".format(args.run_name), exist_ok=True)

    for epoch in range(small_loaded_epoch, small_loaded_epoch + args.epochs):
        
        with tqdm.tqdm(train_dataloader) as loader:
                
            total_loss = 0
            
            for i, (images, labels) in enumerate(loader):
            # for i, (images, labels) in enumerate(train_dataloader):
                if i > args.early_stop:
                    break

                images = images.to(device)
                labels = labels.to(device)

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

                # if i % args.ema_update_freq == 0:
                #     ema_model.update()




            # after epoch, save model checkpoint:
            torch.save(
                small_model.state_dict(),
                "checkpoints/{}/model_{}.pth".format(args.run_name, epoch),
            )



            # after each epoch, sample one batch of images
            with torch.no_grad():
            # with ema_model.average_parameters():
                # break
                # print("sampling")
                # target_label = torch.randint(0, 10, (args.n_samples,)).tolist()
                # images = small_model.sample(20, [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9], return_whole_process=True)
                sample_data = next(iter(test_dataloader))
                input_images =sample_data[0][:args.n_samples].to(device)
                input_labels = torch.Tensor(sample_data[1][:args.n_samples].tolist()) # * 0 # remove labels
                
                if True:
                    reverse_images = small_model.sample(args.n_samples, torch.arange(0, args.n_samples) % 10, keep_intermediate=True)
                    

                    min_max = [(x.min().item(), x.max().item()) for x in reverse_images.unbind(dim=1)]

                    # reverse images shape: (n_samples, markov_states, 1, 14, 14)

                    # now save to GIF with matplotlib animation
                    import matplotlib.pyplot as plt
                    import matplotlib.animation as animation
                    from matplotlib.animation import FuncAnimation
                    import numpy as np

                    frames = torch.unbind(reverse_images, dim=1)
                    # frames has size (n_samples, 1, 14, 14) x markov_states

                    # lay the frames out horizontally
                    frames = [torch.cat(frame.unbind(), dim=-1) for frame in frames]
                    # frames has size (1, 14, 14 * n_samples) x markov_states (1 wide image for each timestep)

                    fig = plt.figure()
                    ims = []
                    for (min, max), i in zip(min_max, range(args.markov_states-1)):
                        im = plt.imshow(frames[i].squeeze(), vmin=min, vmax=max)
                        ims.append([im])
                    for i in range(10): # repeat the last frame 10 times
                        im = plt.imshow(frames[-1].squeeze(), vmin=-1, vmax=1)
                        ims.append([im])
                    
                    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
                    
                    ani.save("images/{}/model_samples_{}.gif".format(args.run_name, epoch), writer=matplotlib.animation.PillowWriter(fps=10))

    
                if True: # show each insta-prediction besides the normal reverse process
                    amount = args.n_samples // 4
                    labels = input_labels[:amount]
                    reverse_images = small_model.sample(amount, labels, keep_intermediate=True)

                    # reverse images is a (amount, markov_states, 1, img_size, img_size) tensor.

                    t = torch.Tensor([i for i in reversed(range(args.markov_states))]).to(device).long()
                    labels = labels.repeat((args.markov_states)).to(device).long()
                    t = t.repeat((amount)).to(device).long()

                    reverse_flattened = torch.flatten(reverse_images, start_dim=0, end_dim=1)
                    insta_predictions = small_model.insta_predict_from_t(reverse_flattened, t, labels)

                    # join the images together along horizontal axis
                    joined = torch.cat((reverse_flattened, insta_predictions), dim=3)
                    

                    # now join the images together along the vertical axis
                    joined = torch.unbind(joined, dim=0)
                    joined = torch.cat((joined), dim=1)
                    joined = torch.chunk(joined, amount, dim=1)
                    joined = torch.stack(joined, dim=0)

                    torchvision.utils.save_image(
                        joined,
                        "images/{}/model_process_{}.png".format(args.run_name, epoch),
                        nrow=amount,
                    )

                # images = insta_predictions

                # print(forward_images.shape)
                # print(reverse_images.shape)

                # rotate the reverse images by 180 degrees
                # reverse_images = torch.flip(reverse_images, dims=[2,3])

                # concat the forward and reverse images:
                # images = torch.cat((forward_images, reverse_images), dim=3)
                # images = torch.cat((forward_images, reverse_images), dim=2)
                #                 # images = forward_images
                # save the images to the run_name path
                # stack the images and the predictions together and save them in one image

                


if __name__ == "__main__":
    args = parse_args()
    main(args)
