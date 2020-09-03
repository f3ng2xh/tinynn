# -*- coding: UTF-8 -*-
import numpy as np


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class SigmoidActivator(object):
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        return 1 * (x > 0)


class ReluActivator(object):
    def forward(self, x):
        return (np.abs(x) + x) / 2

    def backward(self, x):
        return 1 * (x > 0)


class SoftmaxActivator(object):
    def forward(self, x):
        n = np.exp(x)
        d = np.dot(np.ones((1, n.shape[0])), n)
        return n / d

    def backward(self, x):
        return 1 * (x > 0)
