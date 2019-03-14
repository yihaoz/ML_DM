"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y 

    def predict(self, Xtest):
        y_hat = []
        (t, d) = Xtest.shape  # t*d
        dist = utils.euclidean_dist_squared(self.X, Xtest)  # n*t

        # For each row in test
        for i in range(0, t):
            # each col in N*t corresponds one test vector
            row = sorted(dist[:, i])
            row_values = np.array(row[0:self.k])

            row_indices = []
            for j in range(0, self.k):
                row_indices.append(np.where(dist[:, i] == row_values[j]))

            y_kneighbors = []
            for m in row_indices:
                y_kneighbors.append(self.y[m])

            y_hat.append(utils.mode(y_kneighbors))

        return y_hat
