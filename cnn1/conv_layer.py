# -*- coding: UTF-8 -*-
import numpy as np
from scipy.signal import correlate2d
from scipy.signal import convolve2d

"""
z_p^{l+1} = \sum_d w_{d,p}^{l+1} cov a_d^{l} + b_p

w 四维(d,p,u,v)
z 三维(p,j,i)
a 三维(d,h,w)

"""

class ConvLayer(object):
    def __init__(self, input_size, input_dim, zero_padding, stride, kernel_size, n_kernels, activatior):
        self.input_size = input_size
        self.input_dim = input_dim
        self.zero_padding = zero_padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels

        self.output_size = ((input_size - kernel_size + 2 * zero_padding) / stride + 1).astype(int)
        print("input_size: {}, zero_padding: {}, stride: {}, output_size: {}".format(
            repr(input_size), zero_padding, stride, repr(self.output_size)))

        self.weights = np.random.uniform(-1e-4, 1e-4, np.append((input_dim, n_kernels), kernel_size, ))
        self.bias = np.zeros(n_kernels)

        self.activatior = activatior
        self.output_array = None
        self.input_array = None

    def padding_zero(self, input_array, zp):
        if input_array.ndim == 3:
            d, h, w = input_array.shape
            padded_array = np.zeros((d, h + 2 * zp, w + 2 * zp))
            padded_array[:, zp: zp + h, zp: zp + w] = input_array
            return padded_array
        elif input_array.ndim == 2:
            h, w = input_array.shape
            padded_array = np.zeros((h + 2 * zp, w + 2 * zp))
            padded_array[zp: zp + h, zp: zp + w] = input_array
            return padded_array
        raise Exception("wrong padding dim")

    def forward(self, input_array):
        print("input_array:{}".format(input_array.shape))
        padded_array = self.padding_zero(input_array, self.zero_padding)
        print("padded_array:{}".format(repr(padded_array.shape)))

        self.input_array = padded_array

        stride = self.stride

        output_size = np.append(self.n_kernels, self.output_size)
        print("output: {}".format(repr(output_size)))
        self.output_array = np.zeros(output_size)

        for p in range(self.n_kernels):
            z = np.zeros(self.output_size)
            for d in range(self.input_dim):
                w = self.weights[d, p, :, :]
                a = padded_array[d, :, :]
                z += correlate2d(a, w, mode="valid")[::stride, ::stride]  # todo
            z = z + self.bias[p]
            print("z:{}".format(repr(z)))
            self.output_array[p, :, :] = z
        return self.activatior.forward(self.output_array)

    def expand_delta(self, delta):
        depth = delta.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expand_size = self.input_size - self.kernel_size + 2 * self.zero_padding + 1
        print("expand: {}".format(expand_size))
        # 构建新的sensitivity_map
        expand_array = np.zeros(np.append(depth, expand_size))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = delta[:, i, j]
        return expand_array

    """
    dl/dw_{d,p}^{l}=dl/dz_p^{l} cov a_d^{l-1}
    dl/dz_d^{l}=da_d^{l}/dz_d^{l} dot dl/da_d^{l}
    dl/da_d^{l}=\sum_p rot(w_{d,p}^{l+1}) conv dl/dz^{l+1}
    delta.shape=(2,3,3)
    """

    def backward(self, delta):
        expand_delta = self.expand_delta(delta)
        padded_delta = self.padding_zero(expand_delta, self.kernel_size[0] - 1)
        print("delta:{}".format(repr(padded_delta.shape)))
        self.delta = np.zeros(np.append(self.input_dim, self.input_size + 2 * self.zero_padding))
        for d in range(self.input_dim):
            delta1 = np.zeros(self.input_size + 2 * self.zero_padding)
            a = self.input_array[d, :, :]
            for p in range(self.n_kernels):
                # 宽卷积
                w = self.weights[d, p, :, :]
                delta1 += convolve2d(w, padded_delta[p], mode="valid")
            self.delta[d] = np.matmul(a, delta1)

        self.weights_grad = np.zeros(np.append((input_dim, n_kernels), kernel_size, ))
        for d in range(self.input_dim):
            for p in range(self.n_kernels):
                self.weights_grad[d][p] = correlate2d(expand_delta[p], self.input_array[d], mode="valid")
                # print("wg:{}".format(wg.shape))

        self.bias_grad = np.sum(np.sum(delta, axis=2), axis=1)
        print("bias_grad:{}".format(self.bias_grad))

        return self.delta


def test_np():
    input_size = np.array([5, 6])
    kernel_size = np.array([3, 2])

    s = np.array([4, 5])
    d = 1
    print(repr(np.append(s, 1)))
    print(repr(np.zeros(4).shape))

    a = np.array([[1, 2, 0, 0],
                  [5, 3, 0, 4],
                  [0, 0, 0, 7],
                  [9, 3, 0, 0]])
    k = np.array([[1, 1],
                  [0, 1]])
    from scipy.signal import correlate2d
    from scipy.ndimage import convolve

    print(repr(correlate2d(a, k, mode="valid")))
    print(repr(convolve(a, np.rot90(k, 2))))
    print(repr(correlate2d(a, k)))

    print(repr(np.rot90(a, 2)))


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


if __name__ == "__main__":
    input_size = np.array([4, 4])
    input_dim = 3
    zero_padding = 1
    stride = 2
    kernel_size = np.array([2, 2])
    n_kernels = 2

    layer = ConvLayer(input_size, input_dim, zero_padding, stride, kernel_size, n_kernels, IdentityActivator())

    input_array = np.array(
        [[[0, 1, 1, 0],
          [2, 2, 2, 2],
          [1, 0, 0, 2],
          [1, 2, 0, 0]],
         [[1, 0, 2, 2],
          [0, 0, 0, 2],
          [1, 2, 1, 2],
          [1, 2, 1, 1]],
         [[2, 1, 2, 0],
          [1, 0, 0, 1],
          [0, 2, 1, 0],
          [2, 1, 0, 0]]]
    )

    input_array1 = np.array(
        [[[0, 1, 1, 0, 1],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 1],
          [1, 2, 0, 0, 1]],
         [[1, 0, 2, 2, 1],
          [0, 0, 0, 2, 1],
          [1, 2, 1, 2, 1],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 1],
          [1, 0, 0, 1, 1],
          [0, 2, 1, 0, 1],
          [2, 1, 0, 0, 1]]]
    )

    # print("input_array11111:{}".format(repr(input_array1.shape)))

    print(repr(layer.forward(input_array).shape))

    delta = np.ones((2, 3, 3))

    layer.backward(delta)
