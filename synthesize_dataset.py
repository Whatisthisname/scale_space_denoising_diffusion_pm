# load DDPM from checkpoint.
import glob
import re

import torch
from theo_DDPM import DDPM
import argparse

# load latest checkpoint:

def parse_args():
    parser = argparse.ArgumentParser(description="Sampling synthesized MNISTDiffusion dataset")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--size", type=int, default=1000)


    

    args = parser.parse_args()

    # load run_name/args.txt file and parse:
    args_file = "checkpoints/{}/args.txt".format(args.run_name)
    argsDict = {}
    with open(args_file, "r") as f:
        for l in f.readlines():
            k, v = l.split()
            k = k.replace("--", "", 1) # remove first occurence of "--"
            value = v
            try:
                value = eval(v)
            except:
                pass
            argsDict[k] = value

    # overwrite args with args from args.txt:
    for k, v in argsDict.items():
        setattr(args, k, v)
    
    return args


import glob
import re
import os
import shutil
# import cv2
import numpy as np

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPM(args.img_size, ctx_sz=1+10, markov_states=args.markov_states, unet_stages=args.unet_stages, noise_schedule_param=args.noise_power)    

    checkpoints = glob.glob("checkpoints/{}/*.pth".format(args.run_name))
    if len(checkpoints) == 0:
        print(f"No checkpoints found with run name '{args.run_name}', aborting")
        exit()
    else:
        checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
        latest_checkpoint = checkpoints[-1]
        print("Loading checkpoint: {}".format(latest_checkpoint))
        model.load_state_dict(torch.load(latest_checkpoint, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("model loaded.")

    # sample from model:

    # delete files in output directory if it already exists:
    shutil.rmtree("synthesized/{}".format(args.run_name), ignore_errors=True)
    # create output directory:
    os.makedirs("synthesized/{}".format(args.run_name), exist_ok=True)

    batch_size = 100

    images = []
    labels = []
    with torch.no_grad():
        for i in range(args.size // batch_size):
            print(f"Sampling batch {i}")

            gen_labels = torch.randint(0, 10, (batch_size,)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).tolist()

            samples = model.sample(batch_size, target_label=gen_labels, keep_intermediate=False)
            samples = samples.cpu().numpy()

            images.append(samples)
            labels.append(gen_labels)
        
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    np.save(f"synthesized/{args.run_name}/images.npy", images)
    np.save(f"synthesized/{args.run_name}/labels.npy", labels)



if __name__ == "__main__":
    args = parse_args()
    main(args)
