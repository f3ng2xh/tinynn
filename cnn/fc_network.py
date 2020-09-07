# -*- coding: UTF-8 -*-
import numpy as np
from activator import IdentityActivator, ReluActivator, SoftmaxActivator
from fc_layer import FcLayer


class Network(object):
    def __init__(self, input_size, hidden_units, n_class=1):
        self.layers = []
        self.input_size = input_size
        for siz in hidden_units:
            hidden_layer = FcLayer(input_size, siz, activator=ReluActivator())
            self.layers.append(hidden_layer)
            input_size = siz
        self.output_layer = FcLayer(input_size, n_class, SoftmaxActivator())

    def predict(self, all_x):
        preds = []
        for j in range(all_x.shape[0]):
            x = all_x[j]
            pred = self.predict_one_sample(x.reshape(-1, 1))
            preds.append(pred)
        return np.array(preds).reshape(-1, 10)

    def predict_one_sample(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        output = self.output_layer.forward(output)
        return output

    def train_one_sample(self, y, pred, learning_rate):
        # 交叉熵损失函数
        delta = y - pred
        delta = self.output_layer.backward(delta)

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        self.output_layer.update(learning_rate)
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, train_x, train_y, epoch, learning_rate):
        print("start train ...")
        for i in range(epoch):
            for j in range(train_x.shape[0]):
                x, y = train_x[j], train_y[j]
                pred = self.predict_one_sample(x.reshape(-1, 1))
                self.train_one_sample(y.reshape(-1, 1), pred, learning_rate)

        print("train done")

    def loss(self, y_label, y_pred):
        ls = y_label * (-np.log(y_pred))
        return np.sum(ls)


if __name__ == "__main__":
    network = Network(28 * 28, [256, 64, 32], n_class=10)

    input_array = np.random.uniform(0, 1, (2, 28 * 28, 1))
    print("input:{}".format(input_array.shape))
    pred = network.predict_one_sample(input_array[0])
    print("pred:{}".format(pred))

    print("------ train ---------")
    y = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape([2, 10, 1])
    for i in range(100):
        pred = network.predict_one_sample(input_array[0])
        network.train_one_sample(y[0], pred, 0.01)
        ls1 = network.loss(y[0], pred)
        pred = network.predict_one_sample(input_array[1])
        network.train_one_sample(y[1], pred, 0.01)
        ls2 = network.loss(y[1], pred)
        print("loss1:{}".format((ls1 + ls2) / 2))
    print("------ train ---------")

    pred = network.predict_one_sample(input_array[0])
    print("pred0:{}".format(pred))
    pred = network.predict_one_sample(input_array[1])
    print("pred1:{}".format(pred))
