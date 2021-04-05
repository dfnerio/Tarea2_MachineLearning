import LR
import numpy as np

# GRADIENT DESCENT

def gd(X, y, theta, alpha, epochs):
    m = len(X)
    J = [LR.cost(X, y, theta)]
    for i in range(0, epochs):
        h = LR.h(X, theta)
        for i in range(0, len(X.columns)):
            theta[i] -= (alpha/m) * np.sum((h-y)*X.iloc[:, i])
        J.append(LR.cost(X, y, theta))
    return J, theta


def gd_reg(X, y, theta, alpha, epochs):
    m = len(X)
    J = [LR.cost_reg(X, y, theta)]
    for i in range(0, epochs):
        h = LR.h(X, theta)
        for i in range(0, len(X.columns)):
            theta[i] -= (alpha/m) * np.sum((h-y)*X.iloc[:, i])
        J.append(LR.cost_reg(X, y, theta))
    return J, theta


