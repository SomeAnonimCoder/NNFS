from .base_layer import BaseLayer
import numpy as np


class Dropout(BaseLayer):
    """ Dropout layer"""
    def __init__(self, prob=0.5):
        """:param prob: probability of 'disabling'"""
        self.prob = prob

    def fwd(self, input):
        self.mask = np.random.binomial(1, self.prob, size=input.shape) / self.prob
        out = input * self.mask
        return out.reshape(input.shape)

    def backward(self, out_error, lr):
        dX = out_error * self.mask
        return dX, []


