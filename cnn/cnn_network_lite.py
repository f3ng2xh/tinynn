# -*- coding: UTF-8 -*-
import numpy as np
from conv_layer import ConvLayer
from pooling_layer import PoolingLayer
from stacking_layer import StackingLayer
from fc_layer import FcLayer

# from conv_layer import IdentityActivator
from activator import ReluActivator, SoftmaxActivator


class CnnNetwork(object):
    def __init__(self, input_size, n_class=1):
        # input (1,8,8)
        conv1 = ConvLayer(
            input_size=input_size,
            input_dim=1,
            zero_padding=2,
            stride=1,
            kernel_size=np.array([5, 5]),
            n_kernels=3,
            activator=ReluActivator())

        # output (3,8,8)
        self.conv1 = conv1

        pool1 = PoolingLayer(
            input_size=conv1.output_size,
            input_dim=conv1.n_kernels,
            kernel_size=2,
            stride=2,
            mode='max')

        # output (3,4,4)
        self.pool1 = pool1

        stack = StackingLayer(
            input_size=pool1.output_size,
            input_dim=pool1.input_dim)

        # output(3*4*4,1)
        self.stack = stack

        fc1 = FcLayer(
            input_size=stack.output_size,
            output_size=16,
            activator=ReluActivator())

        # output (16,1)
        self.fc1 = fc1

        fc2 = FcLayer(
            input_size=fc1.output_size,
            output_size=n_class,
            activator=SoftmaxActivator())

        # output (10,1)
        self.fc2 = fc2
        self.layers = [conv1, pool1, stack, fc1]
        self.output_layer = fc2

    def predict_one_sample(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        output = self.output_layer.forward(output)

        return output

    def train_one_sample(self, y, pred, learning_rate):
        delta = y - pred
        delta = self.output_layer.backward(delta)

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        self.output_layer.update(learning_rate)
        for layer in self.layers:
            layer.update(learning_rate)

    def loss(self, y_label, y_pred):
        ls = y_label * (-np.log(y_pred))
        return np.sum(ls)


if __name__ == "__main__":
    network = CnnNetwork(input_size=np.array([8, 8]), n_class=10)
    input_array = np.random.uniform(0, 1, (2, 1, 8, 8))
    print("input:{}".format(input_array.shape))
    pred = network.predict_one_sample(input_array[0])
    print("pred:{}".format(pred))

    print("------ train ---------")
    y = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape([2, 10, 1])
    for i in range(10000):
        pred = network.predict_one_sample(input_array[0])
        network.train_one_sample(y[0], pred, 0.001)
        ls1 = network.loss(y[0], pred)
        pred = network.predict_one_sample(input_array[1])
        network.train_one_sample(y[1], pred, 0.001)
        ls2 = network.loss(y[1], pred)
        print("{} - loss1:{}".format(i, (ls1 + ls2) / 2))
    print("------ train ---------")

    pred = network.predict_one_sample(input_array[0])
    print("pred0:{}".format(pred))
    pred = network.predict_one_sample(input_array[1])
    print("pred1:{}".format(pred))
