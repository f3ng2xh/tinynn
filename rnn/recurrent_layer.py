# -*- coding: UTF-8 -*-
import numpy as np

from activator import ReluActivator, IdentityActivator


class RecurrentLayer(object):
    def __init__(self, input_size, state_size, activator):
        self.input_size = input_size
        self.state_size = state_size
        self.activator = activator

        self.w_weights = np.random.uniform(-0.01, 0.01, (self.state_size, self.input_size))
        self.u_weights = np.random.uniform(-0.01, 0.01, (self.state_size, self.state_size))
        self.bias = np.zeros((self.state_size, 1))

        self.reset()

    def reset(self):
        self.times = 0
        self.h_list = []
        self.h_list.append(np.zeros((self.state_size, 1)))

        self.z_list = []
        self.z_list.append(np.zeros((self.state_size, 1)))

        self.delta_list = []
        self.input_list = []
        self.input_list.append(np.zeros((self.input_size, 1)))

        self.w_grad = np.zeros((self.state_size, self.input_size))
        self.u_grad = np.zeros((self.state_size, self.state_size))
        self.b_grad = np.zeros((self.state_size, 1))



    def forward(self, input_array):
        self.times += 1
        z = np.dot(self.u_weights, self.h_list[-1]) + np.dot(self.w_weights, input_array) + self.bias
        self.z_list.append(z)
        self.input_list.append(input_array)
        h = self.activator.forward(z)
        self.h_list.append(h)
        return h

    def backward(self, delta_list_t):
        # delta_list 是 dLt/dZt
        u_grad = np.zeros((self.state_size, self.state_size))
        w_grad = np.zeros((self.state_size, self.input_size))
        b_grad = np.zeros((self.state_size, 1))
        t = self.times
        delta_list_i = delta_list_t
        while t > 0:
            u_grad += np.dot(delta_list_i, self.h_list[t-1].T)
            w_grad += np.dot(delta_list_i, self.input_list[t].T)
            b_grad += delta_list_i
            delta_list_i = self.activator.backward(self.z_list[t-1]) * np.dot(self.u_weights.T, delta_list_i)
            t -= 1

        # t 时刻对 w 导数
        self.w_grad += w_grad
        self.u_grad += u_grad
        self.b_grad += b_grad
        return self.activator.backward(self.input_list[-1]) * np.dot(self.w_weights.T, delta_list_t)

    def update(self, learning_rate):
        self.w_weights += learning_rate * self.w_grad
        self.u_weights += learning_rate * self.u_grad
        self.b_grad += learning_rate * self.b_grad

        self.reset()


def check_gradient():
    error_function = lambda a: a.sum()

    input_array = np.random.uniform(0,1, (3,5,1))
    layer = RecurrentLayer(5, 4, IdentityActivator())

    output = layer.forward(input_array[0])
    print("forward:{}".format(output))

    delta_list = np.ones((4,1), dtype=np.float)
    delta_list1 = layer.backward(delta_list)
    print("w_backward:{}".format(delta_list1))

    print("----------------------------------")

    output = layer.forward(input_array[1])
    print("forward:{}".format(output))

    delta_list = np.ones((4,1), dtype=np.float)
    delta_list1 = layer.backward(delta_list)
    print("w_backward:{}".format(delta_list1))

    # check u_grad
    u_grad = layer.u_grad
    epsilon = 10e-4
    for i in range(4):
        for j in range(4):
            layer.u_weights[i][j] += epsilon
            layer.reset()
            err1 = error_function(layer.forward(input_array[0]))

            layer.u_weights[i][j] -= epsilon * 2
            layer.reset()
            err2 = error_function(layer.forward(input_array[0]))
            expect_grad1 = (err1 - err2) / (2 * epsilon)

            layer.u_weights[i][j] += epsilon * 2
            layer.reset()
            layer.forward(input_array[0])
            err3 = error_function(layer.forward(input_array[1]))

            layer.u_weights[i][j] -= epsilon * 2
            layer.reset()
            layer.forward(input_array[0])
            err4 = error_function(layer.forward(input_array[1]))
            expect_grad2 = (err3 - err4) / (2 * epsilon)

            layer.u_weights[i][j] += epsilon
            print('weights(%d,%d): expected - actural %f - %f' % (i,j, expect_grad1+expect_grad2, u_grad[i][j]))

if __name__ == "__main__":
    check_gradient()

