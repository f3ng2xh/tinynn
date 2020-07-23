# -*- coding: UTF-8 -*-
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivation(x):
    y = sigmoid(x)
    return y * (1 - y)


def relu_derivation(x):
    return 1 * (x > 0)


def relu(x):
    return (np.abs(x) + x) / 2


def softmax(x):
    n = np.exp(x)
    d = np.dot(np.ones((1, n.shape[0])), n)
    return n / d


if __name__ == "__main__":
    x = np.random.randint(-1, 3, size=(2, 1))
    print(x)
    print(relu_derivation(x))
    print(relu(x))

    print(softmax(x))
