
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import csr_matrix as sparse_matrix

# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, \
        item_inverse_mapper, user_ind, item_ind = \
            utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, \
        item_inverse_mapper, user_ind, item_ind = \
            utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # YOUR CODE HERE FOR Q1.1.
        stars = ratings.groupby('item').agg({'rating': 'sum'})
        print(stars[stars['rating'] == stars['rating'].max()])

        # YOUR CODE HERE FOR Q1.1.2
        reviews = ratings.groupby('user').agg({'item': 'count'})
        print(reviews[reviews['item'] == reviews['item'].max()])

        # YOUR CODE HERE FOR Q1.1.3
        # The number of ratings per user
        plt.figure()
        reviews.hist(bins=50)
        plt.yscale('log', nonposy='clip')
        plt.xlabel('Number of ratings per user')
        plt.ylabel('Frequency')
        plt.title('Histogram of the number of ratings per user')
        fname = os.path.join("..", "figs", "q1_1_3_1.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        # The number of ratings per item
        rating_per_item = ratings.groupby('item')['user'].nunique()
        plt.figure()
        rating_per_item.hist(bins=50)
        plt.yscale('log', nonposy='clip')
        plt.xlabel('Number of ratings per item')
        plt.ylabel('Frequency')
        plt.title('Histogram of the number of ratings per item')
        fname = os.path.join("..", "figs", "q1_1_3_2.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        # The rating themselves
        plt.figure()
        ratings.hist(column='rating')
        plt.yscale('log', nonposy='clip')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.title('Histogram of rating')
        fname = os.path.join("..", "figs", "q1_1_3_3.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, \
        item_inverse_mapper, user_ind, item_ind = \
            utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]
        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        # Euclidean distance
        print("For euclidean distance, top similar items are:")

        model = NearestNeighbors()
        model.fit(X.transpose())
        ed_distances, ed_indices = model.kneighbors(grill_brush_vec.transpose(),
                                                    n_neighbors=6)

        for i in np.arange(1, len(ed_indices[0])):
            print(url_amazon % item_inverse_mapper[ed_indices[0, i]])

        # Normalize euclidean distance
        print("For normalize euclidean distance, top similar items are:")
        n = len(set(ratings['user']))
        d = len(set(ratings['item']))
        sum = X.sum(axis=1)
        X_norm = normalize(X, norm='l2', axis=0)

        model.fit(X_norm.transpose())
        grill_brush_vec_norm = normalize(grill_brush_vec,
                                         norm='l2',
                                         axis=0)
        ned_distances, ned_indices = \
            model.kneighbors(grill_brush_vec_norm.transpose(),
                             n_neighbors=6)
        for i in np.arange(1, len(ned_indices[0])):
            print(url_amazon % item_inverse_mapper[ned_indices[0, i]])

        # Cosine similarity
        print("For cosine similarity, top similar items are:")
        model = NearestNeighbors(metric='cosine')
        model.fit(X.transpose())
        cos_distances, cos_indices = \
            model.kneighbors(grill_brush_vec.transpose(),
                             n_neighbors=6)
        for i in np.arange(1, len(cos_indices[0])):
            print(url_amazon % item_inverse_mapper[cos_indices[0, i]])

        # YOUR CODE HERE FOR Q1.3
        # For euclidean distance
        print("For euclidean distance recommended items:")
        for i in np.arange(1, len(ed_distances[0])):
            item_code = item_inverse_mapper[ed_indices[0, i]]
            print("The item {} has {} reviews"
                  .format(item_code,
                          ratings['item'].value_counts().loc[item_code]))

        # For cosine similarity
        print("For cosine similarity recommended items:")
        for i in np.arange(1, len(cos_distances[0])):
            item_code = item_inverse_mapper[cos_indices[0, i]]
            print("The item {} has {} reviews"
                  .format(item_code,
                          ratings['item'].value_counts().loc[item_code]))


    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",
                            filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        len = len(X)
        v = np.full(shape=len, fill_value=1)
        v[400:500] = 0.1
        V = np.diag(v)

        model = linear_model.WeightedLeastSquares()
        model.fit(X, y, V)
        utils.test_and_plot(model,X,y,
                            title="Weighted Least Squares",
                            filename="weight_least_squares_outliers.pdf")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",
                            filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,
                            title="Least Squares, no bias",
                            filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # YOUR CODE HERE
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        utils.test_and_plot(model,X,y,Xtest,ytest,
                            title="Least Squares with bias",
                            filename="least_squares_bias.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)

            # YOUR CODE HERE
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)
            file_name = "least_square_poly_" + str(p)
            title = "Least Squares with polynomial of " + str(p)
            utils.test_and_plot(model,X,y,Xtest,ytest,
                                title=title,filename=file_name)


    else:
        print("Unknown question: %s" % question)

