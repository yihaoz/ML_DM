import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils


# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w


# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,V):
        ''' YOUR CODE HERE '''
        self.w = solve(X.T@V@X, X.T@V@y)


class LinearModelGradient(LeastSquares):
    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w,
                                           lambda w: self.funObj(w,X,y)[0],
                                           epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' %
                  (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):
        # Calculate the function value
        f = np.zeros(len(X[0]))
        for j in range(len(X[0])):
            _sum = 0
            for i in range(len(X)):
                tmp = w.T * X[i].T - y[i]
                _sum += np.sum(np.log(np.exp(tmp) + np.exp(-tmp)))
            f[j] = _sum

        # Calculate the gradient value
        g = np.zeros(len(X[0]))
        for j in range(len(X[0])):
            _sum = 0
            for i in range(len(X)):
                tmp =w.T * X[i].T - y[i]
                _sum += (np.exp(tmp)
                         - np.exp(-tmp))/(np.exp(tmp)
                                          + np.exp(-tmp)) * X[i, j]
            g[j] = _sum
        return f, g


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        w0 = np.ones(shape=(len(X), 1))
        self.Z = np.concatenate((X, w0), axis=1)
        self.w = solve(self.Z.T@self.Z, self.Z.T@y)

    def predict(self, X):
        X = np.concatenate((X, np.ones(shape=(len(X), 1))), axis=1)
        return X@self.w


# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        self.__polyBasis(X)
        self.w = solve(self.Z.T@self.Z, self.Z.T@y)

    def predict(self, X):
        self.__polyBasis(X)
        return self.Z@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        if self.p == 0:
            self.Z = X
            return

        new_cols = np.empty(shape=(len(X), self.p))
        w0 = np.ones(len(X))
        new_cols[:, 0] = w0

        for i in range(2, self.p + 1):
            new_col = np.power(X[:, 0], i)
            new_cols[:, i - 1] = new_col
        self.Z = np.concatenate((X, new_cols), axis=1)
