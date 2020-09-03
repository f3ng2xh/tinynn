# -*- coding: UTF-8 -*-
import numpy as np


class Layer(object):
    def __init__(self, input_size, output_size, activation, derivation):
        self.w = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.activation = activation
        self.derivation = derivation

        self.a = None
        self.z = None
        self.delta = None
        self.w_grad = None
        self.b_grad = None
        self.input = None

    def forward(self, input_value):
        self.input = input_value
        self.z = np.dot(self.w, self.input) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, w, delta):
        if w is None:
            # 最后一层
            self.delta = delta
        else:
            self.delta = self.derivation(self.z) * np.dot(w.T, delta)

        self.w_grad = np.dot(self.delta, self.input.T)
        self.b_grad = self.delta
        return self.delta

    def update(self, alpha):
        self.w = self.w + alpha * self.w_grad
        self.b = self.b + alpha * self.b_grad
