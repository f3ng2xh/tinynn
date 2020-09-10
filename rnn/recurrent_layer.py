# -*- coding: UTF-8 -*-
import numpy as np

from activator import ReluActivator, IdentityActivator


class RecurrentLayer(object):
    def __init__(self, input_size, state_size, max_times, activator):
        self.input_size = input_size
        self.activator = activator

        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((state_size, 1)))

        self.W = np.random.uniform(-0.01, 0.01, (state_size, input_size))
        self.U = np.random.uniform(-0.01, 0.01, (state_size, state_size))
        self.bias = np.zeros((state_size, 1))

        self.delta_list = np.zeros((max_times, max_times, state_size, 1))

    def forward(self, input_array):
        self.times += 1
        current_state = np.dot(self.U, self.state_list[-1]) + np.dot(self.W, input_array) + self.bias
        current_state = self.activator.forward(current_state)
        self.state_list.append(current_state)
        return current_state

    """
    每次 forward 之后, 都要 backward
    """
    def backward(self, delta):
        for t in range(self.times + 1):
            pass

    def bptt(self, delta):
        pass

    def update(self, learning_rate):
        pass


def check_gradient():
    input_array = np.random.uniform(0,1, (10,1))
    layer = RecurrentLayer(10, 16, IdentityActivator())

    print("forward:{}".format(layer.forward(input_array)))

if __name__ == "__main__":
    check_gradient()

