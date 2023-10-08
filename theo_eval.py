from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from torchvision.datasets import MNIST
from torchvision import transforms 

# train random forest classifier on MNIST

def train_rf(images, labels):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(images, labels)
    return clf


def test_rf(clf, images, labels):
    preds = clf.predict(images)
    return accuracy_score(labels, preds)


def compare_classifiers(clf1, clf2, images, labels):
    # create a plot showing classwise accuracy comparison
    preds1 = clf1.predict(images)
    preds2 = clf2.predict(images)
    acc1 = accuracy_score(labels, preds1)
    acc2 = accuracy_score(labels, preds2)
    print("acc1: {}, acc2: {}".format(acc1, acc2))
    classwise_acc1 = []
    classwise_acc2 = []
    for i in range(10):
        classwise_acc1.append(
            accuracy_score(labels[labels == i], preds1[labels == i])
        )
        classwise_acc2.append(
            accuracy_score(labels[labels == i], preds2[labels == i])
        )
    # plot it:
    plt.figure()
    # using bar plots where each bar is a class
    # bars should be side by side for each class
    # x-axis: class
    # y-axis: accuracy
    plt.bar(
        [i - 0.2 for i in range(10)], classwise_acc1, width=0.4, label="clf1", color="r"
    )
    plt.bar(
        [i + 0.2 for i in range(10)], classwise_acc2, width=0.4, label="clf2", color="b"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # load MNIST dataset
    train_data = MNIST(
        root="mnist_data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = MNIST(
        root="mnist_data", train=False, download=True, transform=transforms.ToTensor()
    )

    train_size = 1000
    test_size = 1000
    train_images = train_data.data.reshape(-1, 28 * 28).numpy()[:train_size]
    train_labels = train_data.targets.numpy()[:train_size]
    test_images = test_data.data.reshape(-1, 28 * 28).numpy()[:test_size]
    test_labels = test_data.targets.numpy()[:test_size]

    print("training first:")
    # train random forest classifier on MNIST
    clf = train_rf(train_images, train_labels)
    print("training accuracy: {}".format(test_rf(clf, train_images, train_labels)))
    print("test accuracy: {}".format(test_rf(clf, test_images, test_labels)))

    print("training second:")
    # train random forest classifier on MNIST
    clf2 = train_rf(train_images, train_labels)
    print("training accuracy: {}".format(test_rf(clf2, train_images, train_labels)))
    print("test accuracy: {}".format(test_rf(clf2, test_images, test_labels)))

    compare_classifiers(clf, clf2, test_images, test_labels)
    