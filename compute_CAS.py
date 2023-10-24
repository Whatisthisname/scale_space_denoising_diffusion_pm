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

from models.DDPM import DDPM
from CAS_mnist_classifier import MNISTCLF
from utils import create_mnist_dataloaders

import more_itertools as mit
import itertools as it

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling synthesized MNISTDiffusion dataset")
    # argparse a list of strings:
    parser.add_argument("--run_names", type=str, nargs="+", default=[])

    args = parser.parse_args()

    print(args.run_names)

    return args

# train random forest classifier on MNIST

def train(clf, images, labels):
    optim = torch.optim.AdamW(clf.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epo = 50
    for i in tqdm.tqdm(range(epo), desc="epoch"):
        
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


def plot_classifier_performance(real_clf, test_img, test_label, ax):
    # create a plot showing classwise accuracy comparison

    real_pred = predict(real_clf, test_img)
    real_acc = accuracy_score(test_label, real_pred)

    print("validation set acc: {}".format(real_acc))
    classwise_acc1 = []
    for i in range(10):
        classwise_acc1.append(
            accuracy_score(test_label[test_label == i], real_pred[test_label == i])
        )

    # using bar plots where each bar is a class
    # bars should be side by side for each class
    # x-axis: class
    # y-axis: accuracy

    width = 1/(n_bars+1)
    offset = current_bar_plot * width - 0.5

    ax.bar(
        [i + offset for i in range(10)], classwise_acc1, width=width, label="trained on {}".format(current_dataset)
    )
    


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

    img_size = 28


    #? FIRST, TRAIN A CLASSIC MNIST CLASSIFIER ON THE REAL MNIST DATA AND GET THE VALIDATION-SET ACCURACY

    print("img_size: {}".format(img_size))
    # load MNIST dataset
    preprocess=transforms.Compose([transforms.Resize(img_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0],[1])])
    real_train_data = MNIST(
        root="mnist_data", train=True, download=True, transform=preprocess
    )

    train_size = 10000
    test_size = 1000

    real_train_images = torch.cat([i for (i, l) in it.islice(real_train_data, train_size)], dim=0).unsqueeze(1)
    real_train_labels = real_train_data.targets[:train_size]

    print("training classifier on true data:")
    # train random forest classifier on real MNIST
    real_clf = train(MNISTCLF(img_size), real_train_images, real_train_labels)
    real_train_score = accuracy_score(predict(real_clf, real_train_images), real_train_labels)
    print("train accuracy: {}".format(real_train_score))

    
    validation_data = MNIST(
        root="mnist_data", train=False, download=True, transform=preprocess
    )
    validation_images = torch.cat([i for (i, l) in it.islice(validation_data, test_size)], dim=0).unsqueeze(1)
    validation_labels = validation_data.targets[:test_size]


    ax = plt.subplot(1, 1, 1)
    ax.set_xticks(range(10))
    ax.set_xlabel("class")
    ax.set_ylabel("accuracy")
    # set y bounds to [0, 1]
    ax.set_ylim(0, 1)

    n_bars = len(args.run_names) + 1
    current_bar_plot = 0
    current_dataset = "real"

    plot_classifier_performance(real_clf, validation_images, validation_labels, ax)

    #? THEN, FOR EVERY DDPM-GENERATED DATASET, TRAIN A CLASSIFIER ON THAT DATASET AND GET THE VALIDATION-SET ACCURACY

    for run_name in args.run_names:
        current_bar_plot += 1
        current_dataset = run_name


        print(f"Comparing classifier trained on real MNIST with classifier trained on {run_name}")
        # load synthesized dataset

        fake_train_data = (
            torch.from_numpy(np.load(f"synthesized/{run_name}/images.npy"))
            .reshape(-1, 1, img_size, img_size)
            .float(), 
            torch.from_numpy(np.load(f"synthesized/{run_name}/labels.npy")).long()
        )


        fake_train_images = fake_train_data[0].reshape(-1, img_size * img_size)[:train_size]
        fake_train_labels = fake_train_data[1][:train_size]


        # visualize_images_and_labels(real_train_images, real_train_labels)
        # visualize_images_and_labels(fake_train_imsages, fake_train_labels)
        # visualize_images_and_labels(test_images, test_labels)


        
    
        print("training classifier on fake data from {}:".format(run_name))
        # train random forest classifier on MNIST
        fake_clf = train(MNISTCLF(img_size), fake_train_images, fake_train_labels)
        fake_train_score = accuracy_score(predict(fake_clf, fake_train_images), fake_train_labels)
        print("train accuracy: {}".format(fake_train_score))

        plot_classifier_performance(fake_clf, validation_images, validation_labels, ax)
    
    print("final plotting")
    ax.legend()
    plt.show()
    
    
    