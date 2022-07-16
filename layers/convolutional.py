from .base_layer import BaseLayer
from .utils import init_weights, init_weights_tensor
import numpy as np
import scipy.signal as sp


class Conv(BaseLayer):

    def __init__(self, input_shape, kernel_size):
        self.w = init_weights_tensor([kernel_size]*2)
        self.b = init_weights(*input_shape)

    def backward(self, out_error, lr):
        res = sp.convolve(self.w[::-1], out_error, 'full')
        we = sp.convolve(self.input, out_error,'valid')
        self.b -= out_error * lr
        self.w -= we * lr
        return res

    def fwd(self, input):
        res = sp.convolve(input, self.w, 'same') + self.b
        self.input = input
        return res


"""

    def __init__(self, input_shape, filters, kernel_size):
        self.input_shape=input_shape
        self.w = [init_weights_tensor([kernel_size]*(len(input_shape)-1)+[1]) for _ in range(filters)]
        self.b = [init_weights(*input_shape[1:]) for _ in range(filters)]

    def backward(self, out_error, lr):
        res = np.zeros(self.input_shape)
        for filter,out in zip(self.w, out_error):
            res += sp.convolve(filter[::-1], out, 'full')
        return res

    def fwd(self, input):
        res=[]
        for filter,bias in zip(self.w,self.b):
            res.append(sp.convolve(input, filter, 'same')+bias)
        return res
"""
