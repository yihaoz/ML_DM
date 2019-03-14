# standard Python imports
import os
import argparse
import time
import pickle
import utils
# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
import matplotlib.pyplot as plt
# this comes with Anaconda
from scipy.optimize import approx_fprime        # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # if using Anaconda, install with `conda install scikit-learn`

""" NOTE:
Python is nice, but it's not perfect. One horrible thing about Python is that a 
package might use different names for installation and importing. For example, 
seeing code with `import sklearn` you might sensibly try to install the package 
with `conda install sklearn` or `pip install sklearn`. But, in fact, the actual 
way to install it is `conda install scikit-learn` or `pip install scikit-learn`.
Wouldn't it be lovely if the same name was used in both places, instead of 
`sklearn` and then `scikit-learn`? Please be aware of this annoying feature. 
"""

# CPSC 340 code
import utils
import grads
from decision_stump import DecisionStumpEquality, DecisionStumpErrorRate, DecisionStumpInfoGain
from decision_tree import DecisionTree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "3.3":
        # Here is some code to test your answers to Q3.3
        # Below we test out example_grad using scipy.optimize.approx_fprime, which approximates gradients.
        # if you want, you can use this to test out your foo_grad and bar_grad

        def check_grad(fun, grad):
            x0 = np.random.rand(5) # take a random x-vector just for testing
            diff = approx_fprime(x0, fun, 1e-4)  # don't worry about the 1e-4 for now
            print(x0)
            print("\n** %s **" % fun.__name__)
            print("My gradient     : %s" % grad(x0))
            print("Scipy's gradient: %s" % diff)

        check_grad(grads.example, grads.example_grad)
        check_grad(grads.foo, grads.foo_grad)
        check_grad(grads.bar, grads.bar_grad)


    elif question == "5.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values

        # YOUR CODE HERE
        # 1. The minimum, maximum, mean, median,
        # and mode of all values across the dataset.
        print("The minimum is: %s" % np.ndarray.min(X))
        print("The maximum is: %s" % np.ndarray.max(X))
        print("The mean is: %s" % np.ndarray.mean(X))
        print("The median is: %s" % np.median(X))
        print("The mode is %s" % utils.mode(X))

        # 2. The 5%, 25%, 50%, 75%, and 95% quantiles
        # of all values across the dataset.
        print("The 5%% quantile is %s" % np.percentile(X, 5))
        print("The 25%% quantile is %s" % np.percentile(X, 25))
        print("The 50%% quantile is %s" % np.percentile(X, 50))
        print("The 75%% quantile is %s" % np.percentile(X, 75))
        print("The 95%% quantile is %s" % np.percentile(X, 95))

        # 3. The names of the regions with the highest and lowest means,
        # and the highest and lowest variances.
        regionMean = np.mean(X, axis=0)
        regionVar = np.var(X, axis=0)
        print("The region {} has the highest mean of {} "
              .format(names[np.argmax(regionMean)],np.max(regionMean)))
        print("The region {} has the lowest mean of {} "
              .format(names[np.argmin(regionMean)],np.min(regionMean)))
        print("The region {} has the highest variance of {} "
              .format(names[np.argmax(regionVar)],np.max(regionVar)))
        print("The region {} has the lowest mean of {} "
              .format(names[np.argmin(regionVar)],np.min(regionVar)))

    elif question == "6":
        # 1Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with inequality rule error: %.3f"
              % error)

        # Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q6_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "6.2":
        # Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # Evaluate decision stump
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # Plot result
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        test = DecisionStumpInfoGain()        

    elif question == "6.3":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with info gain rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_3_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
    
    elif question == "6.4":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        model = DecisionTree(max_depth=2,stump_class=DecisionStumpInfoGain)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_4_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "6.5":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)

        min_err_err_rate = 1;
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)

            if min_err_err_rate > my_tree_errors[i]:
                min_err_err_rate = my_tree_errors[i]
                # generate plot
                print("Generating plot")

                utils.plotClassifier(model, X, y)

                fname = os.path.join("..", "figs",
                                     "err_rate_classification.pdf")
                plt.savefig(fname)
                print("\nFigure saved as '%s'" % fname)

        print("Our decision tree with DecisionStumpErrorRate took %f seconds"
              % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="errorrate")

        t = time.time()

        my_tree_errors_infogain = np.zeros(depths.size)

        min_err_info_gain = 1
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth,
                                 stump_class=DecisionStumpInfoGain)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors_infogain[i] = np.mean(y_pred != y)

            if min_err_info_gain > my_tree_errors_infogain[i]:
                min_err_info_gain = my_tree_errors_infogain[i]
                # generate plot
                print("Generating plot")

                utils.plotClassifier(model, X, y)

                fname = os.path.join("..", "figs",
                                     "info_gain_classification.pdf")
                plt.savefig(fname)
                print("\nFigure saved as '%s'" % fname)

        print("Our decision tree with DecisionStumpInfoGain took %f seconds"
              % (time.time()-t))
        
        plt.plot(depths, my_tree_errors_infogain, label="infogain")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)

        min_err_sklearn = 1
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth,
                                           criterion='entropy',
                                           random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn_tree_errors[i] = np.mean(y_pred != y)

            if min_err_sklearn > sklearn_tree_errors[i]:
                min_err_sklearn = sklearn_tree_errors[i]
                # generate plot
                print("Generating plot")

                utils.plotClassifier(model, X, y)

                fname = os.path.join("..", "figs", "sklearn_classification.pdf")
                plt.savefig(fname)
                print("\nFigure saved as '%s'" % fname)

        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, sklearn_tree_errors,
                 label="sklearn", linestyle=":", linewidth=3)

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q6_5_tree_errors.pdf")
        plt.savefig(fname)

    else:
        print("No code to run for question", question)