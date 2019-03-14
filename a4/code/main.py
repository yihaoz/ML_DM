import argparse
import numpy as np
import os
import utils
import linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        W = np.zeros((3, 2))
        W[0,0] = 2
        W[0,1] = -1
        W[1,0] = 2
        W[1,1] = -2
        W[2,0] = 3
        W[2,1] = -1

        print(W)
        print("----------")
        print(np.sum(W[0,:] - W[1,:]))


    if question == "2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL2(lammy=0.9, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("\# nonZeros (features used): %d" % (model.w != 0).sum())

    elif question == "2.2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL1(L1_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("\# nonZeros (features used): %d" % (model.w != 0).sum())

    elif question == "2.3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.5":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # TODO
        regularization = ['l2', 'l1']
        for r in regularization:
            model = LogisticRegression(penalty=r, C=1.0, fit_intercept=False)
            model.fit(XBin, yBin)

            print("scikit-learn {} LogisticRegression Training error {}"
                  .format(r, utils.classification_error(model.predict(XBin), yBin)))
            print("scikit-learn {} LogisticRegression Validation error {}"
                  .format(r, utils.classification_error(model.predict(XBinValid), yBinValid)))
            print("scikit-learn {} LogisticRegression has {} of nonZeros"
                  .format(r, ((model.coef_[0] != 0).sum())))

    elif question == "2.6":
        # w = np.arange(-10, 15, 0.0001)
        w = np.linspace(-1, 1.5, 100000)
        # lam = 1
        # f_1 = 1/2 * np.power(w-2, 2) + 1/2 + lam * np.sqrt(abs(w))
        # plt.plot(w, f_1, 'r--')
        # plt.xlabel("w")
        # plt.ylabel("f(w)")
        # plt.title("f(w) vs w with lambda = 1")
        # fname = os.path.join("..", "figs", "q2_6_lambda_1.pdf")
        # plt.savefig(fname)
        # print("argmin of f(w) is w = {} with lambda = {}".format(w[np.argmin(f_1)], lam))
        lam = 10
        f_10 = 1/2 * np.power(w-2, 2) + 1/2 + lam * np.sqrt(abs(w))
        plt.plot(w, f_10, 'r--')
        plt.xlabel("w")
        plt.ylabel("f(w)")
        plt.title("f(w) vs w with lambda = 10")
        fname = os.path.join("..", "figs", "q2_6_lambda_10.pdf")
        plt.savefig(fname)
        print("argmin of f(w) is w = {} with lambda = {}".format(w[np.argmin(f_10)], lam))


    elif question == "3":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        print(np.unique(model.predict(XMulti)))


    elif question == "3.2":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    elif question == "3.4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    elif question == "3.5":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # TODO
        model = LogisticRegression(C=10000,fit_intercept=False)
        model.fit(XMulti, yMulti)
        print("scikit-learn LogisticRegression One-vs-all:")
        print("Training error %.3f" %
              utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" %
              utils.classification_error(model.predict(XMultiValid), yMultiValid))
        model = LogisticRegression(C=10000,fit_intercept=False,
                                   solver='lbfgs', multi_class='multinomial')
        model.fit(XMulti, yMulti)
        print("scikit-learn LogisticRegression softmax:")
        print("Training error %.3f" %
              utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" %
              utils.classification_error(model.predict(XMultiValid), yMultiValid))