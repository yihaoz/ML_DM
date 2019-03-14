import numpy as np


def predict(X):
    N, D = X.shape
    y = np.zeros(N)

    for n in range(N):
        if X[n, 0] > -81.0:
            if X[n, 1] > 40:
                y[n] = 0
            else:
                y[n] = 1
        else:
            if X[n, 1] > 39:
                y[n] = 0
            else:
                y[n] = 1

    return y