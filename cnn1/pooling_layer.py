# -*- coding: UTF-8 -*-
import numpy as np
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


class PoolingLayer(object):
    def __init__(self, input_size, input_dim, kernel_size, stride, mode='max'):
        self.input_size = input_size
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

        self.output_size = ((input_size - kernel_size) / stride + 1).astype(int)

        self.input_array = None
        self.output_array = None

    def forward(self, input_array):
        self.input_array = input_array
        self.output_array = np.zeros(np.append(self.input_dim, self.output_size))
        for i in range(self.input_dim):
            self.output_array[i] = pool2d(input_array[i, :, :], self.kernel_size, self.stride, pool_mode=self.mode)
        return self.output_array

    def backward(self, delta):
        delta1 = np.zeros(np.append(self.input_dim, self.input_size))
        stride = self.stride

        for d in range(self.input_dim):
            input_strides = self.input_array[d].strides
            stride_array = as_strided(self.input_array[d],
                                      shape=np.append(self.output_size, (self.kernel_size, self.kernel_size)),
                                      strides=(stride * input_strides[0], stride * input_strides[1]) + input_strides)
            print("stride_array:{}".format(stride_array))
            for i in range(self.output_size[0]):
                for j in range(self.output_size[1]):
                    if self.mode == 'avg':
                        delta1[d,
                        (i * stride):(i * stride + stride),
                        (j * stride):(j * stride + stride)] = delta[d, i, j] / (self.kernel_size * self.kernel_size)
                    elif self.mode == 'max':
                        patch_array = stride_array[i, j, :, :]
                        print("stride:{}, argmax:{}".format(patch_array, patch_array.argmax()))
                        k, l = np.unravel_index(patch_array.argmax(), patch_array.shape)
                        print("k:{},l:{}, max:{}".format(k, l, patch_array[k, l]))
                        delta1[d, i * stride + k, j * stride + l] = delta[d, i, j]

        self.delta = delta1
        return delta1


if __name__ == "__main__":
    input_size = np.array([4, 4])
    input_dim = 1
    stride = 2
    kernel_size = 2
    layer = PoolingLayer(input_size, input_dim, kernel_size, stride, mode='avg')

    input_array = np.array([[1, 1, 2, 4],
                            [5, 6, 7, 8],
                            [3, 2, 1, 0],
                            [1, 2, 3, 4]], dtype=np.float).reshape((1, 4, 4))

    print("input:{}".format(input_array))

    output_array = layer.forward(input_array)

    print("output:{}".format(output_array))

    delta = np.ones((1, 2, 2))

    delta1 = layer.backward(delta)
    print("delta1:{}".format(delta1))
