import matplotlib.pyplot as plt
import numpy as np

from datasets.toy_datasets import XorDS
from layers.activation import ActivationTanh
from layers.dense import Dense
from layers.loss import mse, mse_derivative
from network import Network

X, Y, X_test, Y_test = XorDS()
plt.scatter(x=X[:, 0], y=X[:, 1], c=Y)
plt.title('XOR toy dataset')
plt.show()


def show_results(X, Y, nn):
    range = np.arange(-1, 1, 0.01)
    grid = np.array([np.tile(range, len(range)), np.repeat(range, len(range))]).T
    pred = nn.predict(grid)
    plt.imshow(pred.reshape(200, 200), alpha=0.7)
    plt.scatter(x=X[:, 0] * 100 + 100, y=X[:, 1] * 100 + 100, c=Y.round(), edgecolors='grey')
    plt.show()


nn = Network()
nn.add(layer=Dense(2, 5))
nn.add(layer=ActivationTanh())
nn.add(layer=Dense(5, 5))
nn.add(layer=ActivationTanh())
nn.add(layer=Dense(5, 1))
nn.add(layer=ActivationTanh())

nn.set_loss(mse, mse_derivative)
nn.train(X, Y, epochs=40)
show_results(X_test, Y_test, nn)
