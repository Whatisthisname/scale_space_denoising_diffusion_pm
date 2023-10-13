from utils import create_mnist_dataloaders


train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=128, image_size=28
    )

if __name__ == "__main__":
    for i, (images, labels) in (train_dataloader):
        break