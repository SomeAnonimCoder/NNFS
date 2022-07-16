import numpy as np

from .base_layer import BaseLayer


class ActivationTanh(BaseLayer):
    """Activation layer, tanh function"""

    def fwd(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward(self, out_error, lr):
        # for backward propagation we use first derivative of tanh, which is
        # d/dx tanh(x) = 1-tanh^2(x)
        return (1 - np.tanh(self.input) ** 2) * out_error


class ActivationRelu(BaseLayer):
    """Activation layer, ReLU function"""

    def fwd(self, input):
        self.input = input
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, out_error, lr):
        # for backward propagation we use first derivative of relu
        # 1 if x > 0, 0 if x<0
        return (self.input > 0).astype(float) * out_error


class ActivationLeakyRelu(BaseLayer):
    """Activation layer, Leaky ReLU function"""

    def fwd(self, input):
        self.input = input
        self.output = np.maximum(input, 0) + np.minimum(input * 0.01, 0)
        return self.output

    def backward(self, out_error, lr):
        # for backward propagation we use first derivative of LReLu, which is
        # 1 if x > 0, 0.01 if x<0
        return (
            (self.input > 0).astype(float) +
            (self.input < 0).astype(float) * 0.01
               ) * out_error


class ActivationSigmoid(BaseLayer):
    """Activation layer, sigmoid function"""

    def fwd(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, out_error, lr):
        # for backward propagation we use first derivative of sigmoid, which is
        # sigmoid(x)(1-sigmoid(x))
        exponent = np.exp(-self.input)
        return (exponent / (1 + exponent) ** 2) * out_error
