import sys
import GR

# LINEAR REGRESSION

# train

EPOCHS = 15000  # max iterations
THRESHOLD = 0.0000001 # max diff to stop algorithm
DELTA = 0.01  # delta
TH0 = 0  # default value for theta0
TH1 = 0  # default value for theta1

def train(xSet, ySet):

    global TH0
    global TH1

    newTH0, newTH1 = GR.gd(DELTA, THRESHOLD, EPOCHS, xSet, ySet, TH0, TH1)

    TH0 = newTH0
    TH1 = newTH1

    def trained_h(x):
        return h(x, TH0, TH1)

    return trained_h

# test


def test(h, x):
    return h(x)


# helpers

def h(x, th0=TH0, th1=TH1):
    return th0 + th1*x


def cost(xSet, ySet, th0, th1):
    tmp = 0
    for i, x in enumerate(xSet):
        tmp += (h(x, th0, th1) - ySet[i]) ** 2
    return tmp / 2 * (len(xSet))


def costD(xSet, ySet, th0, th1, calc4th1=False):
    tmp = 0
    for i, x in enumerate(xSet):
        if(calc4th1):
            tmp += (h(x, th0, th1) - ySet[i]) * x
        else:
            tmp += (h(x, th0, th1) - ySet[i])
    return tmp / (len(xSet))
