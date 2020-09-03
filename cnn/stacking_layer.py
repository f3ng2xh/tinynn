# -*- coding: UTF-8 -*-
import numpy as np


class StackingLayer(object):
    def __init__(self, input_size, input_dim):
        self.input_size = input_size
        self.input_dim = input_dim
        self.output_size = self.input_size[0] * self.input_size[1] * self.input_dim

    def forward(self, input_array):
        self.output_array = input_array.reshape((-1, 1))
        return self.output_array

    def backward(self, delta):
        self.delta = delta.reshape(np.append(self.input_dim, self.input_size))
        return self.delta

    def update(self, learning_rate):
        pass
