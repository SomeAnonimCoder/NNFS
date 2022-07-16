from datasets.toy_datasets import  XorDS, CirclesDS, WaveDS

import matplotlib.pyplot as plt


X, Y, X_test, Y_test = XorDS()
plt.scatter(x=X[:, 0], y=X[:, 1], c=Y)
plt.title('XOR toy dataset')
plt.show()

X, Y, X_test, Y_test = CirclesDS()
plt.title('Circle toy dataset')
plt.scatter(x=X[:, 0], y=X[:, 1], c=Y)
plt.show()

X, Y, X_test, Y_test = WaveDS()
plt.title('Wave toy dataset')
plt.scatter(x=X[:, 0], y=X[:, 1], c=Y)
plt.show()