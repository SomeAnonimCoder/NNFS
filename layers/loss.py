import numpy as np

def mse(y_true, y):
    """Mean squred error
        :param y: actual output
        :param y_true: desired output
        :return: mse error
    """
    return np.sum((y-y_true)**2)/len(y)

def mse_derivative(y_true, y):
    """Derivative of mean squred error, dE/dY
        :param y: actual output
        :param y_true: desired output
        :return mse derivative, dE/dY
    """
    return 2*(y-y_true)/len(y)