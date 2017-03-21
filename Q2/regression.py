import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import *

def load_data(filename='regression_data.tsv'):
    X = np.loadtxt(filename)
    Y = X[:, -1]
    X = X[:, 0:-1]
    return (X, Y)


def ridge_regression(X, Y, lam=1):
    """ Calculates and returns beta for the ridge regression problem. """
    XT = np.transpose(X)
    I = np.identity(X.shape[1])
    beta = np.dot(np.dot(inv(np.dot(XT,X) + I), XT), Y)
    return beta

def least_squares(X, Y):
    """ Calculates and returns beta for the least squares problem. """
    XT = np.transpose(X)
    beta = np.dot(np.dot(inv(np.dot(XT,X)), XT), Y)
    return beta


def main():
    (X, Y) = load_data()
    rr_estimate = ridge_regression(X, Y)
    ls_estimate = least_squares(X, Y)
    min_bound = min(rr_estimate + ls_estimate)
    max_bound = max(rr_estimate + ls_estimate)
    plt.hist(rr_estimate, bins=50, range=(min_bound, max_bound), color='blue')
    plt.hist(ls_estimate, bins=50, range=(min_bound, max_bound), color='green')
    plt.title(r'$\beta$ Value Histogram for LS and RR Regression')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Number of occurences')
    plt.legend(['Ridge', 'Least-squares'])
    plt.show()

if __name__ == "__main__":
    main()