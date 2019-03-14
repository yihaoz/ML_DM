import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils
from numpy import linalg as LA


class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))
        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)


class logRegL2(logReg):
    # L2 Regularized Logistic Regression
    def __init__(self, lammy=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L2_lambda = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + \
            1/2 * self.L2_lambda * np.square(LA.norm(ord=None, x=w))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.L2_lambda * w

        return f, g


class logRegL1(logReg):
    # L1 Regularized Logistic Regression
    def __init__(self, L1_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L1_lambda = L1_lambda
        self.maxEvals = maxEvals

    def fit(self,X, y):
        n, d = X.shape

        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                        self.maxEvals, X, y, verbose=self.verbose)

class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the selected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature

                X_new = np.empty(shape=(len(X), len(selected_new)))

                counter = 0
                for col in selected_new:
                    X_new[:, counter] = X[:, col]
                    counter += 1

                self.w = np.zeros(len(selected_new))
                (self.w, f) = findMin.findMin(self.funObj, self.w,
                                              self.maxEvals, X_new, y, verbose=self.verbose)

                # add lambda * 0-norm
                if f + self.L0_lambda * (self.w != 0).sum() < minLoss:
                    minLoss = f + self.L0_lambda * (self.w != 0).sum()
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class logLinearClassifier:
    def __init__(self, maxEvals=500, verbose=0):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1
            (self.W[i], f) = findMin.findMin(self.funObj, self.W[i],
                                          self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class softmaxClassifier():
    def __init__(self, maxEvals):
        self.maxEvals = maxEvals
        pass

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # initial guess
        self.W = np.zeros((self.n_classes, d))
        W_vector = self.W.flatten()
        (self.W, f) = findMin.findMin(self.softmaxFun, W_vector,
                                      self.maxEvals, X, y, verbose=0)

        self.W = np.reshape(self.W, (self.n_classes, d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

    def softmaxFun(self, W, X, y):
        n, d = X.shape
        k = self.n_classes

        W = np.reshape(W, (k, d))
        XWT = X.dot(W.T)

        # softmax loss
        f = 0
        for i in range(n):
            sum_exp = 0
            for c in range(k):
                sum_exp += np.exp(XWT[i, c])
            f += -1 * XWT[i, y[i]] + np.log(sum_exp)

        # gradient
        I = np.zeros((n,k))
        p = np.zeros((n,k))

        for i in range(n):
            for c in range(k):
                sum_exp = 0
                for c_prime in range(k):
                    sum_exp += np.exp(XWT[i, c_prime])
                p[i, c] = np.exp(XWT[i, c]) / sum_exp
                if y[i] == c:
                    I[i, c] = 1
        res = p - I
        g = X.T.dot(res).T # X.T.dot(res) refers to Wc,j (d*k) our W is (k*d)

        return f, g.flatten()
