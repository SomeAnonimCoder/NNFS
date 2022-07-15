import numpy as np

from .base_layer import BaseLayer
from .utils import init_weights


class Dense(BaseLayer):
    """Fully-Connected layer, aka Dense in keras"""

    def __init__(self, input_size, output_size):
        """
        Init function
            :param input_size: sise of input, int
            :param output_size: size of output, int
        """
        self.w = init_weights(input_size, output_size)
        self.b = init_weights(1, output_size)

    def fwd(self, input):
        self.input = input
        self.output = np.dot(self.input, self.w) + self.b
        return self.output

    def backward(self, out_error, lr):
        # calculate errors
        input_error = np.dot(out_error, self.w.T)
        weights_error = np.dot(self.input.reshape(1,-1).T, out_error)

        # update params
        self.w -= lr * weights_error
        self.b -= lr * out_error
        return input_error