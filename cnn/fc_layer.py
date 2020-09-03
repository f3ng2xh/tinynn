# -*- coding: UTF-8 -*-
import numpy as np


class FcLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-0.001, 0.001, (output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        self.activator = activator

        self.weights_grad = np.zeros((output_size, input_size))
        self.bias_grad = np.zeros(output_size)

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
        self.weights += learing_rate * self.weights_grad
        self.bias += learing_rate * self.bias_grad
