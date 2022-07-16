from .base_layer import BaseLayer
from .utils import init_weights, init_weights_tensor
import numpy as np
import scipy.signal as sp


class Conv(BaseLayer):

    def __init__(self, input_shape, filters, kernel_size):
        self.filters = filters
        self.w = [init_weights_tensor([kernel_size] * 2) for _ in range(filters)]
        self.b = [init_weights(*input_shape) for _ in range(filters)]

    def backward(self, out_error, lr):
        res = []
        for i in range(self.filters):
            res.append(sp.convolve(self.w[i][::-1], out_error[i], 'full'))
            we = sp.convolve(self.input, out_error[i], 'valid')
            self.b[i] -= out_error[i] * lr
            self.w[i] -= we * lr
        return res

    def fwd(self, input):
        self.input = input
        res = []
        for i in range(self.filters):
            res.append(sp.convolve(input, self.w[i], 'same') + self.b[i])
        return res
