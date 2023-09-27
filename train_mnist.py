import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=2):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    # train data should contain only fives and twos

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )

    train_dataset.data=train_dataset.data[(train_dataset.targets==5)|(train_dataset.targets==2)]
    


    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )
    
    # test_dataset.data=test_dataset.data[(test_dataset.targets==1)|(test_dataset.targets==0)]

    print(test_dataset.data.shape)

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--small_epochs',type = int,default=100)
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=1)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training', default=False)
    parser.add_argument('--run_name',type = str,help = 'define run name', required=True)
    parser.add_argument('--img_size',type = int,help = 'size of image',default='28')

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda"
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=args.img_size)
    
    
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=args.img_size,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.small_epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    
    optimizer=AdamW(model.parameters(),lr=args.lr)
    # scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    global_steps=0


    #load checkpoint
    if args.ckpt:
        # load the latest checkpoint, the one with the biggest steps
        ckpt_list=sorted(os.listdir(f"results/{args.run_name}"),key=lambda x:int(x.split("_")[1].split(".")[0]))
        ckpt=torch.load(f"results/{args.run_name}/{ckpt_list[-1]}")

        global_steps=int(ckpt_list[-1].split("_")[1].split(".")[0])

        model.load_state_dict(ckpt[args.run_name + "_model"])

    else:
        # delete existing train_results/{run_name} folder
        if os.path.exists(f"results/{args.run_name}"):
            os.system(f"rm -rf results/{args.run_name}")


    # train small MNIST model

    for i in range(args.epochs):
        model.train()
        for j,(image,target) in enumerate(train_dataloader):
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            pred=model(image,noise)
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            global_steps+=1
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item())) #,scheduler.get_last_lr()[0]))
        ckpt={args.run_name + "_model":model.state_dict()}

        os.makedirs(f"results/{args.run_name}",exist_ok=True)
        torch.save(ckpt,f"results/{args.run_name}/steps_{global_steps}.pt")

        if args.n_samples>0:
            with torch.no_grad():
                samples = model.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
                save_image(samples,f"results/{args.run_name}/steps_{global_steps}.png",nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args=parse_args()
    main(args)