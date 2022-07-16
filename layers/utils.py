import numpy as np

def init_weights(input_size, output_size):
    """
    Initializer for weights. Currently only normally distributed random
    :param input_size, output_size: size of input and output
    :return weights matrix
    """
    return np.random.normal(0,1,size=[input_size,output_size])


def init_weights_tensor(shape):
    """
    Initializer for weights. Currently only normally distributed random
    :param input_size, output_size: size of input and output
    :return weights matrix
    """
    return np.random.normal(0,1,size=shape)

