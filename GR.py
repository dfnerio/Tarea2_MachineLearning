import LR
import sys

# GRADIENT DESCENT

def gd(a, err, n, xSet, ySet, th0, th1):
    t0 = th0
    t1 = th1

    minCost = sys.maxsize

    for i in range(n):
        tmpCost = LR.cost(xSet, ySet, t0, t1)
        if (tmpCost <= minCost):
            if (minCost - tmpCost <= err):
                break
            minCost = tmpCost
        else:
            break

        temp0 = t0 - a * LR.costD(xSet, ySet, t0, t1)
        temp1 = t1 - a * LR.costD(xSet, ySet, t0, t1, True)
        t0 = temp0
        t1 = temp1

    return t0, t1
