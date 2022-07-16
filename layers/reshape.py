from .base_layer import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def fwd(self, input):
        self.input=input
        return input.reshape(-1)

    def backward(self, out_error, lr):
        return out_error.reshape(self.input.shape)