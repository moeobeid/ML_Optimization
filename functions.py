# IOE 511/MATH 562, University of Michigan
# Code written by: Mouhamad Obeid

# Define all the functions and calculate their gradients, those functions include:
# (1) Linear Least Squares (LLS) function
# (2) Logistic Regression (LR) function

import numpy as np

# Function that computes the function value for the Linear Least Squares function

#           Input: w, X, y
#           Output: F(w)


def LS_func(w, X, y):
    n = X.shape[0]
    return (((X @ w - y).T @ (X @ w - y)) / (2*n))[0][0]

# Function that computes the gradient of the Linear Least Squares function
#
#           Input: w, X, y
#           Output: g = nabla F(w)
#

def LS_grad(w, X, y):
    n = X.shape[0]
    return X.T @ (X @ w - y) / n

# Function that computes the function value for the Logistic Regression function
#
#           Input: w, X, y
#           Output: F(w)
#

def LR_func(w, X, y):
    n = X.shape[0]
    z = X @ w
    return np.sum( np.log(1 + np.exp(-y * z)) ) / n

# Function that computes the gradient of the Logistic Regression function
#
#           Input: w, X, y
#           Output: g = nabla F(w)
#

def LR_grad(w, X, y):
    n = X.shape[0]
    exp = np.exp(-y * (X @ w))
    return -((y * X).T @ (exp / (1 + exp))) / n



