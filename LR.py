import sys
import GR
import numpy as np

# LOGISTIC REGRESSION

# train

EPOCHS = 5000  # max iterations
THRESHOLD = 0.001  # max diff to stop algorithm
ALPHA = 0.0001  # delta
THETA = [0]  # default value for theta

def train(X, Y):

    global THETA
    THETA = [0.5] * len(X.columns)

    print("training...")

    J, THETA = GR.gd(X, Y, THETA, ALPHA, EPOCHS)

    def trained_h(x):
        return h(x, THETA)

    return trained_h

# test

def test(h, x, thres):
    results = h(x)
    for i in range(len(results)):
        if (results[i] >= thres):
            results[i] = 1
        else:
            results[i] = 0
    return results

# helpers

def h(x, theta=THETA):
    return sig(np.dot(theta, x.T)) - 0.0000001

def cost(x, y, theta):
    hip = h(x, theta)
    tmp = 0
    for i in range(len(y)):
        tmp += (y[i]*np.log(hip) + (1-y[i])*np.log(1-hip))
    return -(1/len(x)) * tmp

def sig(x):
    return 1 / (1 + (np.exp(-x)))



