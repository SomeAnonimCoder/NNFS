import numpy as np

from .base_layer import BaseLayer

class ActivationTanh(BaseLayer):
    """Activation layer, tanh only for now"""
    def fwd(self, input):
        self.input=input
        self.output=np.tanh(input)
        return self.output

    def backward(self, out_error, lr):
        # for backward propagation we use first derivative of tanh, which is
        # d/dx tanh(x) = 1-tanh^2(x)
        return (1-np.tanh(self.input)**2)*out_error