# basics
import os
import pickle
import argparse
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depth = np.arange(1, 16)
        tr_error = np.zeros(depth.size)
        te_error = np.zeros(depth.size)

        for i, cur_depth in enumerate(depth):
            model = DecisionTreeClassifier(max_depth=cur_depth,
                                           criterion='entropy',
                                           random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            tr_error[i] = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error[i] = np.mean(y_pred != y_test)

        plt.plot(depth, tr_error, label="Training")
        plt.plot(depth, te_error, label="Test")
        plt.xlabel("Depth of Decision Tree")
        plt.ylabel("Classification Error rate")
        plt.legend()
        fname = os.path.join("..", "figs",
                             "q1_1.pdf")
        plt.savefig(fname)

    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        depth = np.arange(1, 16)
        tr_error = np.ones(depth.size)
        te_error = np.ones(depth.size)

        in_order = True

        if in_order:
            X_train, X_val, y_train, y_val = \
                train_test_split(X,
                                 y,
                                 test_size=0.5,
                                 shuffle=False)
        else:
            X_val, X_train, y_val, y_train = \
                train_test_split(X,
                                 y,
                                 test_size=0.5,
                                 shuffle=False)

        for i, cur_depth in enumerate(depth):
            model = DecisionTreeClassifier(max_depth=cur_depth,
                                           criterion='entropy',
                                           random_state=1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            te_error[i] = np.mean(y_pred != y_val)

        print("minimum training error of {} at depth of {}"
              .format(np.min(te_error), np.argmin(te_error) + 1))
        print(te_error)

    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        print("Column 51 of X is the word '{}'".format(wordlist[50]))
        words = ', '.join(wordlist[np.where(X[500, ] == 1)])
        print("Training example 501 has words: {}".format(words))

        print("Training example 501 comes from newsgroup {}"
              .format(groupnames[y[500]]))

    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("BernoulliNB validation error: %.3f" % v_error)

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        k = [1, 3, 10]
        for k_test in k:
            model = KNN(k_test)
            model.fit(X, y)
            y_pred = model.predict(X)
            y_error = np.mean(y_pred != y)
            print("KNN with k={} gets training error of {}"
                  .format(k_test, y_error))

            y_pred = model.predict(Xtest)
            y_error = np.mean(y_pred != ytest)
            print("KNN with k={} gets validation error of {}"
                  .format(k_test, y_error))

    elif question == '3.3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        k = 1
        model = KNN(k)
        model.fit(X, y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_3_knn_training.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        utils.plotClassifier(model, Xtest, ytest)
        fname = os.path.join("..", "figs", "q3_3_knn_test.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        model = KNeighborsClassifier(n_neighbors=1, p=2)
        model.fit(X, y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_3_sklearn_kneighbors_training.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        utils.plotClassifier(model, Xtest, ytest)
        fname = os.path.join("..", "figs", "q3_3_sklearn_kneighbors_test.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf,
                                    stump_class=DecisionStumpInfoGain))

        print("Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))

        print("Random forest info gain")
        t = time.time()
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))
        print("Our random forest took %f seconds"
              % (time.time()-t))

        print("RandomForestClassifier from sklearn info gain")
        t = time.time()
        evaluate_model(RandomForestClassifier(n_estimators=50,
                                              criterion='entropy',
                                              max_depth=None))
        print("Sklearn RandomForestClassifier took %f seconds"
              % (time.time()-t))

    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        min_error = np.Inf

        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)

            error = model.error(X)
            if (error < min_error):
                min_error = error
                plt.scatter(X[:, 0], X[:, 1], c=model.y, cmap="jet")

                fname = os.path.join("..", "figs", "q5_1.png")
                plt.savefig(fname)
            print("Figure saved as {} with minimum error of {}"
                    .format(fname, min_error))

    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        min_error = np.full(10, np.inf)

        for kk in range(1,11):
            for i in range(50):
                model = Kmeans(k=kk)
                model.fit(X)
                error = model.error(X)
                if error < min_error[kk - 1]:
                    min_error[kk - 1] = error

        plt.plot(min_error)
        plt.xticks(np.arange(0, 10), np.arange(1, 11))
        plt.xlabel("Number of clusters k")
        plt.ylabel("Error")
        plt.title("Minimum error of 50 random initialization vs Number of clusters")

        fname = os.path.join("..", "figs", "q5_2.png")
        plt.savefig(fname)

    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        for epsilon in range(1, 17):
            model = DBSCAN(eps=epsilon, min_samples=3)
            y=model.fit_predict(X)
            if np.unique(model.labels_).size <= 5:
                print("witl ep of {}, it gives clustering of {}"
                      .format(epsilon, np.unique(model.labels_)))
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet", s=5)
                fname = os.path.join("..", "figs", "q5_3_" +
                                     str(np.unique(model.labels_).size - 1) + ".png")
                plt.xlim(-25, 25)
                plt.ylim(-15, 30)
                plt.savefig(fname)
                print("\nFigure saved as '%s'" % fname)
    else:
        print("Unknown question: %s" % question)
