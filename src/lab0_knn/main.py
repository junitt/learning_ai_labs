from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn import Knn
import cv2
from numpy import reshape


def load_mnist(root='./mnist'):

    # TODO Load the MNIST dataset
    # 1. Download the MNIST dataset from
    #    http://yann.lecun.com/exdb/mnist/
    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.

    # Input:
    # root: str, the directory of mnist

    # Output:
    # X_train: np.array, shape (6e4, 28, 28)
    # y_train: np.array, shape (6e4,)
    # X_test: np.array, shape (1e4, 28, 28)
    # y_test: np.array, shape (1e4,)

    # Hint:
    # 1. Use np.fromfile to load the MNIST dataset(notice offset).
    # 2. Use np.reshape to reshape the MNIST dataset.

    # YOUR CODE HERE

    X_train=np.fromfile(root+"\\train-images.idx3-ubyte",dtype=np.uint8,offset=16)
    y_train=np.fromfile(root+"\\train-labels.idx1-ubyte",dtype=np.uint8,offset=8)
    X_test=np.fromfile(root+"\\t10k-images.idx3-ubyte",dtype=np.uint8,offset=16)
    y_test=np.fromfile(root+"\\t10k-labels.idx1-ubyte",dtype=np.uint8,offset=8)
    #print(type(X_test))
    X_train=X_train.reshape(int(6e4),28,28)
    y_train=y_train.reshape(int(6e4),)
    X_test=X_test.reshape(int(1e4),28,28)
    y_test=y_test.reshape(int(1e4),)
    for i in range(len(X_train)):
        _,xx=cv2.threshold(X_train[i],20,255,0)
        X_train[i]=xx
    for i in range(len(X_test)):
        _,xx=cv2.threshold(X_test[i],20,255,0)
        X_test[i]=xx
    #正式测试的时候注释掉下面
    #X_test=X_test[:200]
    #y_test=y_test[:200]
    #在这里考虑进行二值化


    return X_train, y_train, X_test, y_test
    # raise NotImplementedError
    ...

    # End of todo


def main():
    X_train, y_train, X_test, y_test = load_mnist()
    knn = Knn()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()