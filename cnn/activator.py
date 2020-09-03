# -*- coding: UTF-8 -*-
import numpy as np


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class ReluActivator(object):
    def forward(self, x):
        return (np.abs(x) + x) / 2

    def backward(self, y):
        return 1 * (y > 0)


class SoftmaxActivator(object):
    def forward(self, x):
        n = np.exp(x)
        d = np.dot(np.ones((1, n.shape[0])), n)
        return n / d

    def backward(self, y):  # todo
        return 1 * (y > 0)


if __name__ == "__main__":
    ac = ReluActivator()

    x = np.arange(-1, 9).reshape((10, 1))
    print("X:{}".format(x))

    y = ac.forward(x)
    print("y:{}".format(y))

    delta = ac.backward(y)
    print("delta:{}".format(delta))

    print("y>0:{}".format(1 * (y > 0)))
