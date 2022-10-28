# this is for the tree model
from itertools import product
import numpy as np
from pulp import *


def succfailOpt(n, beta, lambd, mu, prevSoln, usePrevSoln, objective):
    m = LpProblem("p", LpMaximize)
    x = LpVariable.dicts("x", (range(n), range(n)), lowBound=0)
    ys = LpVariable.dicts("ys", (range(n), range(n)), lowBound=0)

    m += lpSum([objective[j][i] * x[j][i] for (j, i) in product(range(n), range(n)) if j + i <= n - 1])

    # m = Model()
    # x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
    #       for i in range(n)] for j in range(n)]
    # ys = [[m.add_var(name='slack({},{})'.format(j + 1, i + 1), lb=0)
    #        for i in range(n)] for j in range(n)]
    # m.objective = maximize(xsum(objective[j][i] * x[j][i]
    #                             for j in range(n) for i in range(n) if j + i <= n - 1))

    if usePrevSoln:
        val = 0
        for (j, i) in product(range(n), range(n)):
            if (i > 0) & (j > 0) & (j + i <= n - 1):
                if j + i == n - 1:
                    m += x[j][i] - beta * (j / (i + j + 1)) * prevSoln[j - 1][i] - \
                         beta * (i / (i + j + 1)) * prevSoln[j][i - 1] - beta * prevSoln[j][i] <= 0
                else:
                    m += x[j][i] - beta * (j / (i + j + 1)) * prevSoln[j - 1][i] - \
                         beta * (i / (i + j + 1)) * prevSoln[j][i - 1] <= 0
            elif (i == 0) & (j != 0) & (j + i <= n - 1):
                if j + i == n - 1:
                    m += x[j][i] - beta * (j / (i + j + 1)) * prevSoln[j - 1][i] - beta * prevSoln[j][i] <= 0
                else:
                    m += x[j][i] - beta * (j / (i + j + 1)) * prevSoln[j - 1][i] <= 0
            elif (i != 0) & (j == 0) & (j + i <= n - 1):
                if j + i == n - 1:
                    m += x[j][i] - beta * (i / (i + j + 1)) * prevSoln[j][i - 1] - beta * prevSoln[j][i] <= 0
                else:
                    m += x[j][i] - beta * (i / (i + j + 1)) * prevSoln[j][i - 1] <= 0
            val += objective[j][i] * prevSoln[j][i]
        # uncomment below if want to have a solution not worse than the previous solution
        # m += xsum(objective[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) >= val
    else:
        for (j, i) in product(range(n), range(n)):
            if (i > 0) & (j > 0) & (j + i <= n - 1):
                if j + i == n - 1:
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
                         beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= 0
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
                         beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
                else:
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= 0
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
                         beta * (i / (i + j + 1)) * x[j][i - 1] == - ys[j][i]
                # slack constraints #
                # m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * (i / (i + j + 1)) * x[j][i - 1] \
                #      - eps_slack <= ind[j][i]
                # m += - x[j][i] + beta * (j / (i + j + 1)) * x[j - 1][i] + beta * (i / (i + j + 1)) * x[j][i - 1] \
                #      - eps_slack <= ind[j][i]
                # slack constraints #
            elif (i == 0) & (j != 0) & (j + i <= n - 1):
                if j + i == n - 1:
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * x[j][i] <= 0
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * x[j][i] == - ys[j][i]
                else:
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] <= 0
                    m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] == - ys[j][i]
                # slack constraints #
                # m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - eps_slack <= ind[j][i]
                # m += - x[j][i] + beta * (j / (i + j + 1)) * x[j - 1][i] - eps_slack <= ind[j][i]
                # slack constraints #
            elif (i != 0) & (j == 0) & (j + i <= n - 1):
                if j + i == n - 1:
                    m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= 0
                    m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
                else:
                    m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= 0
                    m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] == - ys[j][i]
                # slack constraints #
                # m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] - eps_slack <= ind[j][i]
                # m += - x[j][i] + beta * (i / (i + j + 1)) * x[j][i - 1] - eps_slack <= ind[j][i]
                # slack constraints #

    m += lpSum([x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1]) <= mu
    m += x[0][0] <= lambd

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
        print("The job constraint is %.10f" % mass)
        print("Objective: ", obj)
        for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
            print("{:<2d} successes and {:<2d} failures have mass "
                  "{:<.18f} and slack {:<.18f}".format(i + 1, j + 1, SolnX[i][j], SolnY[i][j]))

    return SolnX, obj, mass, res


# bounds and slacks will be matrices, in the slacks matrix, the indices with 1 will be forced to have zero slacks
def succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objMult, whichobj):
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
