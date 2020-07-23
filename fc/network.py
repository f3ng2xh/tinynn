# -*- coding: UTF-8 -*-
import numpy as np

from layer import Layer
from activation import *


class Network(object):
    def __init__(self, input_size, hidden_units, n_class=1):
        self.layers = []
        self.input_size = input_size
        for siz in hidden_units:
            hidden_layer = Layer(input_size, siz, activation=relu, derivation=relu_derivation)
            self.layers.append(hidden_layer)
            input_size = siz
        if n_class == 1:
            self.output_layer = Layer(input_size, 1, activation=sigmoid, derivation=None)
        else:
            self.output_layer = Layer(input_size, n_class, activation=softmax, derivation=None)

    def predict(self, all_x):
        preds = []
        for j in range(all_x.shape[0]):
            x = all_x[j]
            pred = self.predict_one(x.reshape(-1,1))
            preds.append(pred)
        return np.array(preds).reshape(-1,10)

    def predict_one(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        output = self.output_layer.forward(output)
        return output

    def train_one_sample(self, y, pred, learning_rate):
        # 交叉熵损失函数
        delta = y - pred
        self.output_layer.backward(None, delta)

        w = self.output_layer.w
        for layer in reversed(self.layers):
            delta = layer.backward(w, delta)
            w = layer.w

        self.output_layer.update(learning_rate)
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, train_x, train_y, epoch, learning_rate):
        print("start train ...")
        for i in range(epoch):
            for j in range(train_x.shape[0]):
                x, y = train_x[j], train_y[j]
                pred = self.predict_one(x.reshape(-1, 1))
                self.train_one_sample(y.reshape(-1, 1), pred, learning_rate)

        print("train done")


if __name__ == "__main__":
    network = Network(5, [4, 3, 2], n_class=10)

    x = np.random.uniform(-1, 1, (5, 1))
    print("x:", x)

    pred = network.predict(x)
    print("predict: ", pred)

    y = np.eye(10)[[1]].T

    print(y)

    for i in range(10):
        network.train_one_step(y, pred, 0.1)

    pred = network.predict(x)
    print("predict: ", pred)

    print(one_hot(1))
