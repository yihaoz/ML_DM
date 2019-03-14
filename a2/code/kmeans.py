import numpy as np
from utils import euclidean_dist_squared
import math

class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        self.means = means

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            # return dist2 of N*k
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                if np.any(y == kk): # don't update the mean if no examples
                    # are assigned to it (one of several possible approaches)
                    means[kk] = X[y == kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'
            # .format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

            error = self.error(X)
            print(error)

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        self.y = np.argmin(dist2, axis=1)
        return self.y

    def error(self, X):
        N, D = X.shape
        y_pred = self.predict(X)
        approx_val = np.zeros((N, D))

        for i in range(N):
            approx_val[i] = self.means[y_pred[i]]

        error = 0
        for i in range(N):
            for j in range(D):
                error += math.pow((approx_val[i, j] - X[i, j]), 2)

        return error
