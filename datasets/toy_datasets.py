import numpy as np

PI = 3.14


def XorDS():
    X = np.random.uniform(-1, 1, [1000, 2])
    Y = (X[:, 0] * X[:, 1] > 0).astype(float)

    X_test = np.random.uniform(-1, 1, [300, 2])
    Y_test = (X_test[:, 0] * X_test[:, 1] > 0).astype(float)
    return X, Y, X_test, Y_test


def CirclesDS():
    X = np.random.uniform(-1, 1, [1000, 2])
    Y = np.sin((X[:, 0] ** 2 + X[:, 1] ** 2) * 2 * PI)
    X_test = np.random.uniform(-1, 1, [300, 2])
    Y_test = np.sin((X_test[:, 0] ** 2 + X_test[:, 1] ** 2) * 2 * PI)
    return X, Y, X_test, Y_test


def WaveDS():
    X = np.random.uniform(-1, 1, [1000, 2])
    x = X[:, 0]
    y = X[:, 1]
    Y = np.sin(x * PI) * np.sin(y * PI)

    X_test = np.random.uniform(-1, 1, [300, 2])
    x = X_test[:, 0]
    y = X_test[:, 1]
    Y_test = np.sin(x * PI) * np.sin(y * PI)
    return X, Y, X_test, Y_test
