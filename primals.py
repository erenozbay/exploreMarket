# this is for the tree model
from itertools import product
import numpy as np
from pulp import *


# bounds and slacks will be matrices, in the slacks matrix, the indices with 1 will be forced to have zero slacks
def succfailOpt(n, beta, lambd, mu, prevSoln, usePrevSoln, objective, tr, slacks,
                bounds=np.ones(1), whichobj='OPT'):  # whichobj argument, if in {-1, 0, 1}, chooses the fixed point obj
    m = LpProblem("p", LpMaximize)
    x = LpVariable.dicts("x", (range(n), range(n)), lowBound=0)
    ys = LpVariable.dicts("ys", (range(n), range(n)), lowBound=0)
    y = LpVariable("y")

    if whichobj != 'OPT':
        if whichobj == 1:
            m += y
        elif whichobj == -1:
            m += -y
            m += y >= 0
        elif whichobj == 0:
            m += lpSum([-objective[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1])
    else:
        m += lpSum([objective[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1])

    if usePrevSoln:
        val = 0
        for (j, i) in product(range(n), range(n)):
            ind = j + i == n - 1  # if at leaf node
            s = tr[str(j - 1) + str(i)]['s'] if j > 0 else 0  # success to state (j, i)
            f = tr[str(j) + str(i - 1)]['f'] if i > 0 else 0  # failure to state (j, i)
            r = 1 - tr[str(j) + str(i)]['r'] if i + j < n - 1 else 1  # remain
            # in form (1 - prob) to shorten the constraint variable
            if (i > 0) & (j > 0):
                m += x[j][i] * r - beta * s * prevSoln[j - 1][i] - beta * f * prevSoln[j][i - 1] - \
                     beta * prevSoln[j][i] * ind <= 0
                m += x[j][i] * r - beta * s * prevSoln[j - 1][i] - beta * f * prevSoln[j][i - 1] - \
                     beta * prevSoln[j][i] * ind == - ys[j][i]
            elif (i == 0) & (j != 0):
                m += x[j][i] * r - beta * s * prevSoln[j - 1][i] - beta * prevSoln[j][i] * ind <= 0
                m += x[j][i] * r - beta * s * prevSoln[j - 1][i] - beta * prevSoln[j][i] * ind == - ys[j][i]
            elif (i != 0) & (j == 0):
                m += x[j][i] * r - beta * f * prevSoln[j][i - 1] - beta * prevSoln[j][i] * ind <= 0
                m += x[j][i] * r - beta * f * prevSoln[j][i - 1] - beta * prevSoln[j][i] * ind == - ys[j][i]
            val += objective[j][i] * prevSoln[j][i]
        # uncomment below if want to have a solution not worse than the previous solution
        # m += xsum(objective[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) >= val
    else:
        for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j < n):
            ind = j + i == n - 1  # if at leaf node
            s = tr[str(j - 1) + str(i)]['s'] if j > 0 else 0  # success to state (j, i)
            f = tr[str(j) + str(i - 1)]['f'] if i > 0 else 0  # failure to state (j, i)
            r = 1 - tr[str(j) + str(i)]['r'] if i + j < n - 1 else 1  # remain
            # in form (1 - prob) to shorten the constraint variable
            if (i > 0) & (j > 0):
                m += x[j][i] * r - beta * s * x[j - 1][i] - beta * f * x[j][i - 1] - beta * x[j][i] * ind <= 0
                m += x[j][i] * r - beta * s * x[j - 1][i] - beta * f * x[j][i - 1] - beta * x[j][i] * ind == - ys[j][i]
            elif (i == 0) & (j != 0):
                m += x[j][i] * r - beta * s * x[j - 1][i] - beta * x[j][i] * ind <= 0
                m += x[j][i] * r - beta * s * x[j - 1][i] - beta * x[j][i] * ind == - ys[j][i]
            elif (i != 0) & (j == 0):
                m += x[j][i] * r - beta * f * x[j][i - 1] - beta * x[j][i] * ind <= 0
                m += x[j][i] * r - beta * f * x[j][i - 1] - beta * x[j][i] * ind == - ys[j][i]

    m += lpSum([x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1]) <= mu
    if whichobj != 'OPT':
        m += lpSum([x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1]) == mu

    m += x[0][0] <= lambd
    m += x[0][0] - lambd == - ys[0][0]

    if whichobj == 'OPT':
        for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
            m += ys[i][j] <= slacks[i][j]
    else:
        for (i, j) in product(range(n), range(n)):
            m += x[i][j] <= bounds[i][j]
        for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
            m += ys[i][j] >= y
            if slacks[i][j] == 1:
                m += ys[i][j] == 0

    res = not m.solve(PULP_CBC_CMD(msg=False)) == LpSolutionOptimal

    SolnX = np.zeros((n, n))
    SolnY = np.zeros((n, n))
    mass = 0
    obj = 0
    for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
        SolnX[j][i] = value(x[j][i])
        SolnY[j][i] = value(ys[j][i])
        mass += value(x[j][i])
        obj += objective[j][i] * value(x[j][i])

    if not res:
        if whichobj == 'OPT':
            print("The job constraint is %.10f" % mass)
            print("Objective: ", obj)
            for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
                print("{:<2d} successes and {:<2d} failures have mass "
                      "{:<.18f} and slack {:<.18f}".format(i + 1, j + 1, SolnX[i][j], SolnY[i][j]))
        else:
            if whichobj == 1:
                print("regular objective")
            elif whichobj == -1:
                print("other objective")
            elif whichobj == 0:
                print("minimize the rewards objective")
            print("Objective is ", obj, ", and y is ", value(y))

    if whichobj == 'OPT':
        return SolnX, res, obj, mass
    else:
        return SolnX, res, SolnY


# bounds and slacks will be matrices, in the slacks matrix, the indices with 1 will be forced to have zero slacks
def DEPRECATED_succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objMult, whichobj):
    m = LpProblem("p", LpMaximize)
    x = LpVariable.dicts("x", (range(n), range(n)), lowBound=0)
    y = LpVariable("y")

    if whichobj == 1:
        m += y
    elif whichobj == -1:
        m += -y
        m += y >= 0
    elif whichobj == 0:
        m += lpSum([-objMult[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1])

    for (i, j) in ((i, j) for j in range(n) for i in range(n) if (j + i <= n - 1) & (slacks[i][j] == 1)):
        if (i + j) == 0:
            m += x[0][0] >= lambd
        elif (i == 0) & (j != 0):
            if j == n - 1:
                m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] >= 0
            else:
                m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] >= 0
        elif (i != 0) & (j == 0):
            if i == n - 1:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * x[i][j] >= 0
            else:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] >= 0
        else:
            if j + i == n - 1:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - \
                     beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] >= 0
            else:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - \
                     beta * (j / (i + j + 1)) * x[i][j - 1] >= 0

    for (i, j) in product(range(n), range(n)):
        m += x[i][j] <= bounds[i][j]
        if (i > 0) & (j > 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - \
                     beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] <= 0
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - \
                     beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] <= -y
            else:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * (j / (i + j + 1)) * x[i][j - 1] <= 0
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * (j / (i + j + 1)) * x[i][j - 1] <= -y
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * x[i][j] <= 0
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * x[i][j] <= -y
            else:
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] <= 0
                m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] <= -y
        elif (i == 0) & (j != 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] <= 0
                m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] <= -y
            else:
                m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] <= 0
                m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] <= -y

    m += lpSum([x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1]) == mu
    m += x[0][0] <= lambd

    res = not m.solve(PULP_CBC_CMD(msg=False)) == LpSolutionOptimal

    soln_sub = np.zeros((n, n))
    obj = 0
    for (j, i) in product(range(n), range(n)):
        soln_sub[j][i] = value(x[j][i])
        obj += objMult[j][i] * soln_sub[j][i]
    if not res:
        if whichobj == 1:
            print("regular objective")
        elif whichobj == -1:
            print("other objective")
        elif whichobj == 0:
            print("minimize the rewards objective")
        print("Objective is ", obj, ", and y is ", value(y))
    # else:
    #     print("Infeasible")
    return soln_sub, res
