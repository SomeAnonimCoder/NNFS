import numpy as np


def mse(y_true, y):
    """Mean squred error
        :param y: actual output
        :param y_true: desired output
        :return: mse error
    """
    return np.sum((y - y_true) ** 2) / len(y)


def mse_derivative(y_true, y):
    """Derivative of mean squred error, dE/dY
        :param y: actual output
        :param y_true: desired output
        :return mse derivative, dE/dY
    """
    return 2 * (y - y_true) / len(y)


def mae(y_true, y):
    """Mean absolute error loss
    :param y: actual output
    :param y_true: desired output
    :return: mae error"""
    return np.sum(np.abs(y - y_true))


def mae_derivative(y_true, y):
    """Mean absolute error derivative
    :param y: actual output
    :param y_true: desired output
    :return: mae derivative"""
    delta = y - y_true
    return (delta > 0).astype(float) - (delta < 0).astype(float)


def mape(y_true, y):
    """Mean absolute percentage error loss
    :param y: actual output
    :param y_true: desired output
    :return: mape error"""
    return np.sum(np.abs((y - y_true) / y_true))


def mape_derivative(y_true, y):
    """Mean absolute percentage error derivative
    :param y: actual output
    :param y_true: desired output
    :return: mape derivative"""
    delta = y - y_true
    return (delta > 0).astype(float) / np.abs(y_true) - (delta < 0).astype(float) / np.abs(y_true)


def _softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def cross_entropy(labels, logits):
    """
    Cross-entropy loss
    :param logits: output from neural net, shape is (num_examples,num_classes)
    :param labels: labels from y_train. NB! They must not be one-hot encoded!
    :return error
    """
    p = _softmax(logits)
    log_likelihood = -np.log(p[0, labels])
    loss = np.sum(log_likelihood)
    return loss


def cross_entropy_derivative(y, X):
    """
      Cross-entropy loss derivative
      :param logits: output from neural net, shape is (num_examples,num_classes)
      :param labels: labels from y_train. NB! They must not be one-hot encoded!
      :return error gradient
      """
    grad = _softmax(X)
    grad[0, y] -= 1
    grad = grad
    return grad
