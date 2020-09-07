# -*- coding: UTF-8 -*-
import numpy as np
from activator import IdentityActivator


class FcLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        self.activator = activator

        self.weights_grad = None
        self.bias_grad = None

        self.input_array = None
        self.output_array = None
        self.delta = None

    def forward(self, input_array):
        self.input_array = input_array
        self.output_array = np.dot(self.weights, self.input_array) + self.bias
        return self.activator.forward(self.output_array)

    def backward(self, delta):
        self.delta = self.activator.backward(self.input_array) * np.dot(self.weights.T, delta)
        self.weights_grad = np.dot(delta, self.input_array.T)
        self.bias_grad = delta

        return self.delta

    def update(self, learing_rate):
        self.weights = self.weights + learing_rate * self.weights_grad
        self.bias = self.bias + learing_rate * self.bias_grad


def check_gradient():
    error_function = lambda a: a.sum()

    input_size = 8
    output_size = 2

    layer = FcLayer(input_size, output_size, IdentityActivator())

    input_array = np.array([[2, 1, 4, 5, 6, 7, 9, 1]], dtype=np.float64).reshape((8, 1))

    layer.forward(input_array)
    delta = np.ones(((2, 1)), dtype=np.float64)
    delta = layer.backward(delta)

    # 检查 weights
    layer.forward(input_array)
    delta = np.ones(((2, 1)), dtype=np.float64)
    delta = layer.backward(delta)

    epsilon = 10e-4

    for i in range(input_size):
        input_array[i, 0] += epsilon
        output_array = layer.forward(input_array)
        err1 = error_function(output_array)
        input_array[i, 0] -= epsilon * 2
        err2 = error_function(layer.forward(input_array))
        expect_grad = (err1 - err2) / (2 * epsilon)
        input_array[i, 0] += epsilon
        print('input (%d): expected - actural %f - %f' % (
            i, expect_grad, delta[i]))

    for i in range(output_size):
        for j in range(input_size):
            layer.weights[i, j] += epsilon
            err1 = error_function(layer.forward(input_array))
            layer.weights[i, j] -= epsilon * 2
            err2 = error_function(layer.forward(input_array))
            expect_grad = (err1 - err2) / (2 * epsilon)
            layer.weights[i, j] += epsilon
            print('weights(%d,%d): expected - actural %f - %f' % (
                i, j, expect_grad, layer.weights_grad[i, j]))

    # 检查 bias
    for p in range(output_size):
        layer.bias[p, 0] += epsilon
        err1 = error_function(layer.forward(input_array))
        layer.bias[p, 0] -= epsilon * 2
        err2 = error_function(layer.forward(input_array))
        expect_grad = (err1 - err2) / (2 * epsilon)
        layer.bias[p, 0] += epsilon
        print('bias(%d): expected - actural %f - %f' % (
            p, expect_grad, layer.bias_grad[p, 0]))


if __name__ == "__main__":
    check_gradient()
