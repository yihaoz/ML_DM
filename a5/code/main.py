import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2
from pca import PCA, AlternativePCA, RobustPCA

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # standard logistic regression
        lr = logRegL2(lammy=1)
        lr.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr.predict(Xtest) != ytest))

        utils.plotClassifier(lr, Xtrain, ytrain)
        utils.savefig("logReg.png")
        
        # kernel logistic regression with a linear kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_linear, lammy=0.01)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # YOUR CODE HERE
        # Polynomial kernel
        poly_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_poly, lammy=0.01)
        poly_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(poly_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(poly_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(poly_kernel, Xtrain, ytrain)
        utils.savefig("PolynomialKernel.png")

        # kernel RBF
        rbf_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=0.01, sigma=0.5)
        rbf_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(rbf_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(rbf_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(rbf_kernel, Xtrain, ytrain)
        utils.savefig("RBFKernel.png")

    elif question == "1.2":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # YOUR CODE HERE
        # kernel RBF
        min_err_train, min_err_val = 1, 1
        sigma_min_err_train, lam_min_err_val = 0, 0
        sigma_min_err_val, lam_min_err_val = 0, 0

        sigmas = np.float_power(10, np.arange(-2, 3, 1))
        lams = np.float_power(10, np.arange(-4, 1, 1))
        for lam in lams:
            rbf_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=lam, sigma=0.5)
            rbf_kernel.fit(Xtrain, ytrain)
            err_train = np.mean(rbf_kernel.predict(Xtrain) != ytrain)
            err_val = np.mean(rbf_kernel.predict(Xtest) != ytest)
            print("training error is {} with sigma={}"
                  .format(err_train, lam))
            print("validation error is {} with sigma={}"
                  .format(err_val, lam))

        # for sigma in sigmas:
        #     for lam in lams:
        #         rbf_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=lam, sigma=sigma)
        #         rbf_kernel.fit(Xtrain, ytrain)
        #         err_train = np.mean(rbf_kernel.predict(Xtrain) != ytrain)
        #         err_val = np.mean(rbf_kernel.predict(Xtest) != ytest)
        #
        #         if (err_train < min_err_train):
        #             print("New minimum training error is {} with sigma={} and lambda={}"
        #                   .format(err_train, sigma, lam))
        #             min_err_train = err_train
        #         if (err_val < min_err_val):
        #             print("New minimum validation error is {} with sigma={} and lambda={}"
        #                   .format(err_val, sigma, lam))
        #             min_err_val = err_val
                # print("training error is {} with sigma={} and lambda={}"
                #       .format(err_train, sigma, lam))
                # print("validation error is {} with sigma={} and lambda={}"
                #       .format(err_val, sigma, lam))

    elif question == '4.1': 
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n,d = X.shape
        print(n,d)
        h,w = 64,64      # height and width of each image

        k = 5        # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray', vmin=0, vmax=1)

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray', vmin=0, vmax=1)

            utils.savefig('highway_{:03d}.jpg'.format(i))

    else:
        print("Unknown question: %s" % question)    