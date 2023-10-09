from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from torchvision.datasets import MNIST
from torchvision import transforms 

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
import argparse
import torch_ema as ema
import tqdm

from theo_DDPM import DDPM
from theo_mnist_classifier import MNISTCLF
from train_mnist import create_mnist_dataloaders

import more_itertools as mit
import itertools as it


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling synthesized MNISTDiffusion dataset")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dataset_size", type=int, default=1000)

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

# train random forest classifier on MNIST

def train(clf, images, labels):
    optim = torch.optim.AdamW(clf.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epo = 50
    for i in range(epo):
        epoch_total_loss = 0
        for (images, labels) in zip(mit.chunked(images, 16), mit.chunked(labels, 16)):
            images = list ( map (lambda x: torch.Tensor(x).reshape(1, 1, img_size, img_size), images) )
            labels = torch.Tensor(labels).long()
            images = torch.concat(images, dim=0)

            optim.zero_grad()
            preds = clf(images)

            loss = criterion(preds, labels)
            loss.backward()
            optim.step()
            epoch_total_loss += loss.item()

        averaged = epoch_total_loss / (len(labels) // 16)
        print(f"\repoch {i+1:2}/{epo} loss: {averaged:.4f}", end="")
    print()
    return clf


def predict(clf, images):
    with torch.no_grad():
        preds = clf(torch.Tensor(images).reshape(-1, 1, img_size, img_size)).argmax(dim=1).numpy()
    return preds


def compare_classifiers(real_clf, fake_clf, test_img, test_label):
    # create a plot showing classwise accuracy comparison
    fake_pred = predict(fake_clf, test_img)
    real_pred = predict(real_clf, test_img)
    real_acc = accuracy_score(test_label, real_pred)
    fake_acc = accuracy_score(test_label, fake_pred)
    print("On test set:\nreal acc: {}, fake acc: {}".format(real_acc, fake_acc))
    classwise_acc1 = []
    classwise_acc2 = []
    for i in range(10):
        classwise_acc1.append(
            accuracy_score(test_label[test_label == i], real_pred[test_label == i])
        )
        classwise_acc2.append(
            accuracy_score(test_label[test_label == i], fake_pred[test_label == i])
        )
    # plot it:
    plt.figure()
    # using bar plots where each bar is a class
    # bars should be side by side for each class
    # x-axis: class
    # y-axis: accuracy
    plt.bar(
        [i - 0.2 for i in range(10)], classwise_acc1, width=0.4, label="trained on real", color="r"
    )
    plt.bar(
        [i + 0.2 for i in range(10)], classwise_acc2, width=0.4, label="trained on fake", color="b"
    )
    plt.xticks(range(10))
    plt.xlabel("class")
    plt.legend()
    plt.show()


def visualize_images_and_labels(images, labels):
    # visualize some images with their labels
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(img_size, img_size), cmap="gray")
        plt.title(labels[i])
    plt.show()

if __name__ == "__main__":
    args = parse_args()

    img_size = args.img_size
    print("img_size: {}".format(img_size))
    # load MNIST dataset
    preprocess=transforms.Compose([transforms.Resize(args.img_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0],[1])])
    real_train_data = MNIST(
        root="mnist_data", train=True, download=True, transform=preprocess
    )
    fake_train_data = (
        torch.from_numpy(np.load(f"synthesized/{args.run_name}/images.npy"))
        .reshape(-1, 1, args.img_size, args.img_size)
        .float(), 
        torch.from_numpy(np.load(f"synthesized/{args.run_name}/labels.npy")).long()
    )
    
    test_data = MNIST(
        root="mnist_data", train=False, download=True, transform=preprocess
    )


    train_size = 10000
    test_size = 1000

    real_train_images = torch.cat([i for (i, l) in it.islice(real_train_data, train_size)], dim=0).unsqueeze(1)
    real_train_labels = real_train_data.targets[:train_size]
    
    fake_train_images = fake_train_data[0].reshape(-1, img_size * img_size)[:train_size]
    fake_train_labels = fake_train_data[1][:train_size]

    test_images = torch.cat([i for (i, l) in it.islice(test_data, test_size)], dim=0).unsqueeze(1)
    test_labels = test_data.targets[:test_size]

    # visualize_images_and_labels(real_train_images, real_train_labels)
    visualize_images_and_labels(fake_train_images, fake_train_labels)
    # visualize_images_and_labels(test_images, test_labels)


    print("training classifier on true data:")
    # train random forest classifier on real MNIST
    real_clf = train(MNISTCLF(args.img_size), real_train_images, real_train_labels)
    real_train_score = accuracy_score(predict(real_clf, real_train_images), real_train_labels)
    print("train accuracy: {}".format(real_train_score))
    

    print("training classifier on fake data:")
    # train random forest classifier on MNIST
    fake_clf = train(MNISTCLF(args.img_size), fake_train_images, fake_train_labels)
    fake_train_score = accuracy_score(predict(fake_clf, fake_train_images), fake_train_labels)
    print("train accuracy: {}".format(fake_train_score))

    compare_classifiers(real_clf, fake_clf, test_images, test_labels)
    