from itertools import product
from mip import *
from plotnine import *
import numpy as np
import pandas as pd
import random
# from copy import deepcopy
import time
# import warnings
import statistics


# import matplotlib.pyplot as plt

# direct dual of the tree model
def succfailDual(n, beta, lambd, mu, objective):
    m = Model()
    g = [[m.add_var(name='gamma({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    alpha = m.add_var(lb=0)
    m.objective = minimize(lambd * g[0][0] + alpha * mu)

    for (j, i) in product(range(n), range(n)):
        if i + j < n - 1:
            m += alpha >= beta * ((j + 1) / (i + j + 2)) * g[j + 1][i] + \
                 beta * ((i + 1) / (i + j + 2)) * g[j][i + 1] + objective[j][i] - g[j][i]
        else:
            m += alpha >= g[j][i] * (beta - 1) + objective[j][i]

    m.optimize()
    print('Objective value is ', m.objective_value)
    soln = np.zeros((n, n))
    print(alpha.x)
    for (i, j) in product(range(n), range(n)):
        soln[i][j] = g[i][j].x
    print(soln)
    return soln


# dual of the tree model using a fixed point
def succfailDualUseFixedPoint(n, beta, lambd, mu, x, objective):
    rhs = np.zeros((n, n))
    rhs[0][0] = lambd
    for (j, i) in product(range(n), range(n)):
        if (i > 0) & (j > 0) & (j + i <= n - 1):
            if j + i == n - 1:
                rhs[j][i] = beta * (j / (i + j + 1)) * x[j - 1][i] + \
                            beta * (i / (i + j + 1)) * x[j][i - 1] + beta * x[j][i]
            else:
                rhs[j][i] = beta * (j / (i + j + 1)) * x[j - 1][i] + \
                            beta * (i / (i + j + 1)) * x[j][i - 1]
        elif (i == 0) & (j != 0) & (j + i <= n - 1):
            if j + i == n - 1:
                rhs[j][i] = beta * (j / (i + j + 1)) * x[j - 1][i] + beta * x[j][i]
            else:
                rhs[j][i] = beta * (j / (i + j + 1)) * x[j - 1][i]
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if j + i == n - 1:
                rhs[j][i] = beta * (i / (i + j + 1)) * x[j][i - 1] + beta * x[j][i]
            else:
                rhs[j][i] = beta * (i / (i + j + 1)) * x[j][i - 1]
    m = Model()
    g = [[m.add_var(name='gamma({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    alpha = m.add_var(lb=0)
    m.objective = minimize(xsum(rhs[i][j] * g[i][j] for i in range(n) for j in range(n)) + alpha * mu)

    for (i, j) in product(range(n), range(n)):
        m += g[i][j] + alpha >= objective[i][j]

    m.optimize()
    print('Objective value is ', m.objective_value)
    soln = np.zeros((n, n))
    print(alpha.x)
    for (i, j) in product(range(n), range(n)):
        soln[i][j] = g[i][j].x
    print(soln)
    return soln


def succfailOptPriors(n, beta, lambd, mu, objective):
    m = Model()
    x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    ys = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
           for i in range(n)] for j in range(n)]
    m.objective = maximize(xsum(objective[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1))

    for (j, i) in product(range(n), range(n)):
        if (i > 0) & (j > 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
            else:
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - beta * (1 - objective[j][i - 1]) * x[j][
                    i - 1] <= 0
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - objective[j][i - 1]) * x[j][i - 1] == - ys[j][i]
        elif (i == 0) & (j != 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - beta * x[j][i] <= 0
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - beta * x[j][i] == - ys[j][i]
            else:
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] <= 0
                m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] == - ys[j][i]
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
            else:
                m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] <= 0
                m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] == - ys[j][i]

    m += xsum(x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) <= mu

    m += x[0][0] <= lambd
    m.optimize()
    jobcon = 0
    for (j, i) in product(range(n), range(n)):
        jobcon += x[j][i].x

    if m.optimize() != OptimizationStatus.INFEASIBLE:
        print("The job constraint is %.10f" % xsum(x[j][i].x for j in range(n) for i in range(n)))
        print("Objective: ", m.objective_value)
        for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
            print("{:<2d} successes and {:<2d} failures have mass "
                  "{:<.18f} and slack {:<.18f}".format(i + 1, j + 1, x[i][j].x, ys[i][j].x))

    return m.objective_value, jobcon


def succfailOpt(n, beta, lambd, mu, prevSoln, usePrevSoln, objective):
    m = Model()
    x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    ys = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
           for i in range(n)] for j in range(n)]
    m.objective = maximize(xsum(objective[j][i] * x[j][i]
                                for j in range(n) for i in range(n) if j + i <= n - 1))

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

    m += xsum(x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) <= mu

    m += x[0][0] <= lambd
    m.optimize()
    newSoln = np.zeros((n, n))
    jobcon = 0
    for (j, i) in product(range(n), range(n)):
        newSoln[j][i] = x[j][i].x
        jobcon += x[j][i].x

    if m.optimize() != OptimizationStatus.INFEASIBLE:
        print("The job constraint is %.10f" % xsum(x[j][i].x for j in range(n) for i in range(n)))
        print("Objective: ", m.objective_value)
        for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
            print("{:<2d} successes and {:<2d} failures have mass "
                  "{:<.18f} and slack {:<.18f}".format(i + 1, j + 1, x[i][j].x, ys[i][j].x))

    return newSoln, m.objective_value, jobcon


pd.set_option('display.max_columns', None)


# bounds and slacks will be matrices, in the slacks matrix, the indices with 1 will be forced to have zero slacks
def succfailOptFixedPoint(n, beta, lambd, mu, bounds, slacks, objMult, whichobj):
    m = Model()
    x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    y = m.add_var()
    if whichobj == 1:
        print("regular objective")
        m.objective = maximize(y)
    else:
        print("other objective")
        m.objective = minimize(y)
        m += y >= 0

    if bounds.sum() > mu:
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if (j + i <= n - 1) & (slacks[i][j] == 1)):
            if (i + j) == 0:
                m += x[0][0] >= lambd
                # m += y[i][j] <= lambd - x[0][0]
            elif (i == 0) & (j != 0):
                if j + i == n - 1:
                    m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] >= 0
                    # m += y[i][j] >= beta * (j / (i + j + 1)) * x[i][j - 1] + beta * x[i][j] - x[i][j]
                else:
                    m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] >= 0
                    # m += y[i][j] >= beta * (j / (i + j + 1)) * x[i][j - 1] - x[i][j]
            elif (i != 0) & (j == 0):
                if j + i == n - 1:
                    m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * x[i][j] >= 0
                    # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] + beta * x[i][j] - x[i][j]
                else:
                    m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] >= 0
                    # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] - x[i][j]
            else:
                if j + i == n - 1:
                    m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - \
                         beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] >= 0
                    # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] + \
                    #      beta * (j / (i + j + 1)) * x[i][j - 1] + beta * x[i][j] - x[i][j]
                else:
                    m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * (j / (i + j + 1)) * x[i][
                        j - 1] >= 0

    for (j, i) in product(range(n), range(n)):
        m += x[j][i] <= bounds[j][i]
        if (i > 0) & (j > 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
                     beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
                     beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= 0
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
                     beta * (i / (i + j + 1)) * x[j][i - 1] <= -y
        elif (i == 0) & (j != 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * x[j][i] <= 0
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] <= 0
                m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] <= -y
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= 0
                m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= -y

    m += xsum(x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) == mu

    m += x[0][0] <= lambd
    m.optimize()

    soln_sub = np.zeros((n, n))
    obj = 0
    for (j, i) in product(range(n), range(n)):
        soln_sub[j][i] = x[j][i].x
        obj += objMult[j][i] * soln_sub[j][i]
    print("Objective is ", obj, ", and y is ", y.x)
    return soln_sub, m.optimize()


def succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objMult, whichobj):
    m = Model()
    x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    y = m.add_var()
    if whichobj == 1:
        print("regular objective")
        m.objective = maximize(y)
    elif whichobj == -1:
        print("other objective")
        m.objective = minimize(y)
        m += y >= 0
    else:
        print("minimize the rewards objective")
        m.objective = minimize(xsum(objMult[j][i] * x[j][i]
                                for j in range(n) for i in range(n) if j + i <= n - 1))

    if bounds.sum() > mu:
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if (j + i <= n - 1) & (slacks[i][j] == 1)):
            if (i + j) == 0:
                m += x[0][0] >= lambd
                # m += y[i][j] <= lambd - x[0][0]
            elif (i == 0) & (j != 0):
                if j + i == n - 1:
                    m += x[i][j] - beta * (1 - objMult[i][j - 1]) * x[i][j - 1] - beta * x[i][j] >= 0
                    # m += y[i][j] >= beta * (j / (i + j + 1)) * x[i][j - 1] + beta * x[i][j] - x[i][j]
                else:
                    m += x[i][j] - beta * (1 - objMult[i][j - 1]) * x[i][j - 1] >= 0
                    # m += y[i][j] >= beta * (j / (i + j + 1)) * x[i][j - 1] - x[i][j]
            elif (i != 0) & (j == 0):
                if j + i == n - 1:
                    m += x[i][j] - beta * objMult[i - 1][j] * x[i - 1][j] - beta * x[i][j] >= 0
                    # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] + beta * x[i][j] - x[i][j]
                else:
                    m += x[i][j] - beta * objMult[i - 1][j] * x[i - 1][j] >= 0
                    # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] - x[i][j]
            else:
                if j + i == n - 1:
                    m += x[i][j] - beta * objMult[i - 1][j] * x[i - 1][j] - \
                         beta * (1 - objMult[i][j - 1]) * x[i][j - 1] - beta * x[i][j] >= 0
                    # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] + \
                    #      beta * (j / (i + j + 1)) * x[i][j - 1] + beta * x[i][j] - x[i][j]
                else:
                    m += x[i][j] - beta * objMult[i - 1][j] * x[i - 1][j] - \
                         beta * (1 - objMult[i][j - 1]) * x[i][j - 1] >= 0

    for (j, i) in product(range(n), range(n)):
        m += x[j][i] <= bounds[j][i]
        if (i > 0) & (j > 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - objMult[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - objMult[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - objMult[j][i - 1]) * x[j][i - 1] <= 0
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - objMult[j][i - 1]) * x[j][i - 1] <= -y
        elif (i == 0) & (j != 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] - beta * x[j][i] <= 0
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] <= 0
                m += x[j][i] - beta * objMult[j - 1][i] * x[j - 1][i] <= -y
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * (1 - objMult[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * (1 - objMult[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * (1 - objMult[j][i - 1]) * x[j][i - 1] <= 0
                m += x[j][i] - beta * (1 - objMult[j][i - 1]) * x[j][i - 1] <= -y

    m += xsum(x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) == mu

    m += x[0][0] <= lambd
    m.optimize()

    soln_sub = np.zeros((n, n))
    obj = 0
    for (j, i) in product(range(n), range(n)):
        soln_sub[j][i] = x[j][i].x
        obj += objMult[j][i] * soln_sub[j][i]
    print("Objective is ", obj, ", and y is ", y.x)
    return soln_sub, m.optimize()


def succfailFixedPointPriors(n, beta, lambd, mu, objective):
    # print out the fixed point whenever you find it
    slacks, bounds = np.zeros((n, n)), np.zeros((n, n))
    # (0, 0) should be positive, hence the slacks for all states strictly better than (0, 0) should be zero
    bounds[0][0] = mu
    minrew = 1
    for (i, j) in ((i, j) for j in range(n) for i in range(n) if (i + j <= n - 1)):
        if minrew >= objective[i][j]:
            minrew = objective[i][j]
        if objective[i][j] > objective[0][0]:
            slacks[i][j] = 1
            bounds[i][j] = mu
    print("call model")
    print(bounds)
    # last argument 1 implies objective is maximize(a), last argument 0 implies objective is minimize(a)
    a_soln, feasibility = succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objective, 1)
    a_soln_b, feasibility_b = succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objective, 0)
    objectivevalue = 0
    objectivevalue_b = 0
    finalobjval = 0
    changed_soln = 0
    both_solns = -1
    if feasibility == OptimizationStatus.OPTIMAL:
        fp = a_soln
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectivevalue += objective[i][j] * fp[i][j]
    if feasibility_b == OptimizationStatus.OPTIMAL:
        fp_b = a_soln_b
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectivevalue_b += objective[i][j] * fp_b[i][j]
    if (objectivevalue > 0) & (objectivevalue_b > 0):
        keepsoln = fp if objectivevalue_b > objectivevalue else fp_b
        print(keepsoln)
        both_solns = 1
        finalobjval -= finalobjval
        finalobjval = min(objectivevalue, objectivevalue_b)
        if abs(objectivevalue - objectivevalue_b) > 1e-6:
            changed_soln += 1
    elif objectivevalue > 0:
        both_solns = 0
        keepsoln = fp
        print(keepsoln)
        finalobjval -= finalobjval
        finalobjval = objectivevalue
    elif objectivevalue_b > 0:
        both_solns = 0
        keepsoln = fp_b
        print(keepsoln)
        finalobjval -= finalobjval
        finalobjval = objectivevalue_b

    success_prob = objective[0][0]  # the success probability of (0, 0)
    counter = 0
    while (success_prob >= minrew) and (counter < n * n * 0.6):
        counter += 1
        nextsuccess_prob = 0
        ind = -1
        possible_indices = np.zeros(((n + 1) // 2, 2))
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] <= success_prob)
                                                                      & (i + j <= n - 1))):
            if objective[i][j] >= success_prob:
                ind += 1
                possible_indices[ind][0] = i
                possible_indices[ind][1] = j
            else:
                if objective[i][j] >= nextsuccess_prob:  # to get the next success_prob I can use
                    nextsuccess_prob = objective[i][j]
        if (possible_indices > 0).any():  # if you get no options above, directly move on to the next success_prob
            # enumerate all possible solutions with the slack choices
            # print(possible_indices)
            if ind > 0:
                for s in product([0, 1], repeat=(ind + 1)):
                    if sum(s) > 0:
                        # print(s)
                        # print(bounds)
                        # print()
                        for inc in range(ind + 1):
                            ii = int(possible_indices[inc][0])
                            jj = int(possible_indices[inc][1])
                            if s[inc] == 1:
                                bounds[ii][jj] = mu
                            else:
                                bounds[ii][jj] = 0
                        # print(bounds)
                        # a_soln_prev = np.zeros((n, n))
                        # for k in range(5):
                        #     multip = [[np.random.binomial(1, 0.5) for _ in range(n)] for _ in range(n)]
                        #     multip[1][1] = 10 * multip[1][1]
                        #     multip[2][0] = 3 * multip[2][0]
                        #     print(multip)
                        objectivevalue = 0
                        firsttime = 0
                        for jk in range(2):
                            # print("call with multiple choices")
                            a_soln, feasibility = succfailOptFixedPointPriors(n, beta, lambd, mu, bounds,
                                                                              slacks, objective, jk)
                            if (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 0):
                                keepsoln = a_soln
                                for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                                    objectivevalue += objective[i][j] * a_soln[i][j]
                                # print(a_soln)
                                finalobjval -= finalobjval
                                finalobjval = objectivevalue
                                firsttime = 1
                            elif (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 1):
                                changed_soln += 1
                                objectivevalue_b = 0
                                for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                                    objectivevalue_b += objective[i][j] * a_soln[i][j]
                                if objectivevalue_b < objectivevalue:
                                    finalobjval -= finalobjval
                                    finalobjval = objectivevalue_b
                                    keepsoln = a_soln
                                    # print(a_soln)

            else:
                # print(bounds)
                ii = int(possible_indices[ind][0])
                jj = int(possible_indices[ind][1])
                bounds[ii][jj] = mu
                # print(bounds)
                objectivevalue = 0
                firsttime = 0
                for jk in range(2):
                    # print("call with single choice")
                    a_soln, feasibility = succfailOptFixedPointPriors(n, beta, lambd, mu, bounds,
                                                                      slacks, objective, jk)
                    if (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 0):
                        keepsoln = a_soln
                        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                            objectivevalue += objective[i][j] * a_soln[i][j]
                        # print(a_soln)
                        finalobjval -= finalobjval
                        finalobjval = objectivevalue
                        firsttime = 1
                    elif (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 1):
                        objectivevalue_b = 0
                        changed_soln += 1
                        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                            objectivevalue_b += objective[i][j] * a_soln[i][j]
                        if objectivevalue_b < objectivevalue:
                            finalobjval -= finalobjval
                            finalobjval = objectivevalue_b
                            keepsoln = a_soln
                            # print(a_soln)

        success_prob = nextsuccess_prob
        # before moving on to the next success_prob, I need to fix the slacks and bounds for all those with higher probs
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] > success_prob)
                                                                      & (i + j <= n - 1))):
            slacks[i][j] = 1
            bounds[i][j] = mu

    print("\n Final objval in fixed point is ", finalobjval)
    return keepsoln, finalobjval, changed_soln, both_solns


def succfailFixedPoint(n, beta, lambd, mu, objective):
    # print out the fixed point whenever you find it
    slacks, bounds = np.zeros((n, n)), np.zeros((n, n))
    # (0, 0) should be positive, hence the slacks for all states strictly better than (0, 0) should be zero
    bounds[0][0] = mu
    minrew = 1
    for (i, j) in ((i, j) for j in range(n) for i in range(n) if (i + j <= n - 1)):
        if minrew >= objective[i][j]:
            minrew = objective[i][j]
        if objective[i][j] > objective[0][0]:
            slacks[i][j] = 1
            bounds[i][j] = mu
    print("call model")
    # print(bounds)
    a_soln, feasibility = succfailOptFixedPoint(n, beta, lambd, mu, bounds, slacks, objective, 1)
    if feasibility == OptimizationStatus.OPTIMAL:
        fp = a_soln
        objectivevalue = 0
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectivevalue += objective[i][j] * fp[i][j]
        print(a_soln)

    success_prob = objective[0][0]  # the success probability of (0, 0)
    counter = 0
    while (success_prob >= minrew) and (counter < n * n * 0.6):
        counter += 1
        nextsuccess_prob = 0
        ind = -1
        possible_indices = np.zeros(((n + 1) // 2, 2))
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] <= success_prob)
                                                                      & (i + j <= n - 1))):
            if objective[i][j] >= success_prob:
                ind += 1
                possible_indices[ind][0] = i
                possible_indices[ind][1] = j
            else:
                if objective[i][j] >= nextsuccess_prob:  # to get the next success_prob I can use
                    nextsuccess_prob = objective[i][j]
        if (possible_indices > 0).any():  # if you get no options above, directly move on to the next success_prob
            # enumerate all possible solutions with the slack choices
            # print(possible_indices)
            if ind > 0:
                for s in product([0, 1], repeat=(ind + 1)):
                    if sum(s) > 0:
                        # print(s)
                        # print(bounds)
                        # print()
                        for inc in range(ind + 1):
                            ii = int(possible_indices[inc][0])
                            jj = int(possible_indices[inc][1])
                            if s[inc] == 1:
                                bounds[ii][jj] = mu
                            else:
                                bounds[ii][jj] = 0
                        # print(bounds)
                        for jk in range(2):
                            print("call with multiple choices")
                            a_soln, feasibility = succfailOptFixedPoint(n, beta, lambd, mu, bounds,
                                                                        slacks, objective, jk)
                            if feasibility == OptimizationStatus.OPTIMAL:
                                fp = a_soln
                                objectivevalue = 0
                                for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                                    objectivevalue += objective[i][j] * fp[i][j]
                                print(a_soln)
            else:
                # print(bounds)
                ii = int(possible_indices[ind][0])
                jj = int(possible_indices[ind][1])
                bounds[ii][jj] = mu
                # print(bounds)
                for jk in range(2):
                    print("call with single choice")
                    a_soln, feasibility = succfailOptFixedPoint(n, beta, lambd, mu, bounds, slacks,
                                                                objective, jk)
                    if feasibility == OptimizationStatus.OPTIMAL:
                        fp = a_soln
                        objectivevalue = 0
                        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                            objectivevalue += objective[i][j] * fp[i][j]
                        print(a_soln)

        success_prob = nextsuccess_prob
        # before moving on to the next success_prob, I need to fix the slacks and bounds for all those with higher probs
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] > success_prob)
                                                                      & (i + j <= n - 1))):
            slacks[i][j] = 1
            bounds[i][j] = mu

    return fp, objectivevalue


def succfailSim(state, T, workerarriveprob, jobarriveprob, wsp, bigK, rewardprob, C, percent):#, iii, jjj):
    counter_conv, total_reward, counterr = 0, 0, 0
    queue, track_assign, queue_mid = np.zeros((state, state)), np.zeros((state, state)), np.zeros((state, state))
    track_mass, track_queues, track_queues_cum = np.zeros((int(T / 10), int((state + 1) * state * 0.5) + 1)), \
                                                 np.zeros((int(T / 10), int((state + 1) * state * 0.5) + 1)), \
                                                 np.zeros((int(T / 10), int((state + 1) * state * 0.5) + 1))
    last_queues = np.zeros((int(T * (1 - percent / 100)), int((state + 1) * state * 0.5) + 1))
    pricesHere = np.zeros((state, state))
    ####

    workerarrival = np.random.binomial(1, (np.ones(T) * workerarriveprob))  # vector of arrivals for workers
    # jobarrival = np.ones(T) * jobarriveprob #+ np.random.binomial(1, (np.ones(T) * 0.1))
    jobarrival = np.random.binomial(1, (np.ones(T) * jobarriveprob))  # vector of arrivals for jobs
    print("total arrivals will be ", jobarrival.sum())

    for t in range(T):
        queue[0][0] += workerarrival[t]
        if (jobarrival[t] >= 1) & (queue.sum() > 0):  # to assign, I need a job and the system non-empty
            maxval = 0
            randomize = 0

            ################ new steps ################
            # priceadjmatrix = np.ones((state, state)) * (-C)
            # # first two cols position, next is number in queue, last is price adjusted reward
            # # allstates = np.zeros((int(state * (state + 1) / 2), 4))
            # fourstates = np.zeros((4, 2))
            # pos_i = np.zeros(4)
            # pos_j = np.zeros(4)
            # maxvals = np.zeros(4)
            # for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if (i + j <= (state - 1)) &
            #                                                                            (queue[i][j] > 0)):
            #     priceadjmatrix[i][j] = rewardprob[i][j] - C * (bigK - min(bigK, queue[i][j])) / bigK
            #     if maxval <= priceadjmatrix[i][j]:
            #         maxval = priceadjmatrix[i][j]
            #         if maxval >= maxvals[0]:
            #             pos_i[1], pos_i[2], pos_i[3] = pos_i[0], pos_i[1], pos_i[2]
            #             pos_j[1], pos_j[2], pos_j[3] = pos_j[0], pos_j[1], pos_j[2]
            #             pos_i[0], pos_j[0] = i, j
            #             maxvals[0], maxvals[1], maxvals[2], maxvals[3] = maxval, maxvals[0], maxvals[1], maxvals[2]
            #         elif (maxval < maxvals[0]) & (maxval >= maxvals[1]):
            #             pos_i[2], pos_i[3] = pos_i[1], pos_i[2]
            #             pos_j[2], pos_j[3] = pos_j[1], pos_j[2]
            #             pos_i[1], pos_j[1] = i, j
            #             maxvals[1], maxvals[2], maxvals[3] = maxval, maxvals[1], maxvals[2]
            #         elif (maxval < maxvals[1]) & (maxval >= maxvals[2]):
            #             pos_i[3] = pos_i[2]
            #             pos_j[3] = pos_j[2]
            #             pos_i[2], pos_j[2] = i, j
            #             maxvals[2], maxvals[3] = maxval, maxvals[2]
            #         elif (maxval < maxvals[2]) & (maxval >= maxvals[3]):
            #             pos_i[3], pos_j[3] = i, j
            #             maxvals[3] = maxval
            # for jj in range(4):
            #     fourstates[jj][0], fourstates[jj][1] = pos_i[jj], pos_j[jj]
            #     # fourstates[jj][2], fourstates[jj][3] = queue[pos_i[jj]][pos_j[jj]], priceadjmatrix[pos_i[jj]][pos_j[jj]]
            # # print("at time ", t, " states ", fourstates, " and queues ", queue, " price ", priceadjmatrix)
            # doassignment = (maxvals[0] > 0) * queue[int(pos_i[0])][int(pos_j[0])] + \
            #                (maxvals[1] > 0) * queue[int(pos_i[1])][int(pos_j[1])] + \
            #                (maxvals[2] > 0) * queue[int(pos_i[2])][int(pos_j[2])] + \
            #                (maxvals[3] > 0) * queue[int(pos_i[2])][int(pos_j[3])]
            # if doassignment > 0:
            #     tot_assigned = 0
            #     indexx = -1
            #     tobeassigned = jobarrival[t]
            #     while (tot_assigned < tobeassigned):  # & (indexx < 3):
            #         indexx += 1
            #         pos_ii, pos_jj = int(fourstates[indexx][0]), int(fourstates[indexx][1])
            #         # print(pos_ii, " ", pos_jj)
            #         losing = int(tobeassigned) if tobeassigned <= queue[pos_ii][pos_jj] else int(queue[pos_ii][pos_jj])
            #         # print("losing ", losing, " at time ", t)
            #         tot_assigned += losing
            #         tobeassigned -= losing
            #         # print(indexx, " ", tobeassigned)
            #         queue[pos_ii][pos_jj] -= losing
            #         track_assign[pos_ii][pos_jj] += losing
            #         for jkj in range(losing):
            #             reward_param = np.random.beta(pos_ii + iii + 1, pos_jj + jjj + 1)
            #             reward = np.random.binomial(1, reward_param)
            #             if t > T * percent / 100:
            #                 total_reward += reward
            #             stay = np.random.binomial(1, wsp)  # see if it will stay
            #
            #             if stay == 1:  # if the assigned worker is staying
            #                 if pos_ii + pos_jj < (state - 1):
            #                     queue[pos_ii + 1][pos_jj] += reward
            #                     queue[pos_ii][pos_jj + 1] += (1 - reward)
            #                 else:
            #                     queue[pos_ii][pos_jj] += 1
            ################ new steps ################

            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                # if i + j < (state - 1):
                #     if (i + j == 0) or (i * j >= 1):
                #         price = (bigK - min(bigK, queue[i][j])) / bigK - \
                #                     wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, queue[i + 1][j])) / bigK - \
                #                     wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, queue[i][j + 1])) / bigK
                #     elif i == 0:
                #         price = (bigK - min(bigK, queue[i][j])) / bigK - \
                #                 wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, queue[i][j + 1])) / bigK
                #     elif j == 0:
                #         price = (bigK - min(bigK, queue[i][j])) / bigK - \
                #                 wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, queue[i + 1][j])) / bigK
                # else:
                price = (bigK - min(bigK, queue[i][j])) / bigK
                if (maxval <= (rewardprob[i][j] - C * price)) & (queue[i][j] > 0):
                    if (maxval < (rewardprob[i][j] - C * price)) & (
                            (rewardprob[i][j] - C * price) > 0):  # to randomize selections of, e.g., (1,1) and (2,2)
                        randomize = 1
                        maxval = rewardprob[i][j] - C * price
                        pos_i = i
                        pos_j = j
                    elif randomize > 0:
                        if rewardprob[i][j] == rewardprob[pos_i][pos_j]:
                            choose = random.uniform(0, 1)
                            if choose >= 0.5:
                                pos_i = i
                                pos_j = j
            if maxval > 0:
                if (queue < 0).any():
                    print("oops, a non-existent worker left.")
                    break
                queue[pos_i][pos_j] -= 1
                track_assign[pos_i][pos_j] += 1
                # reward_param = np.random.beta(pos_i + iii + 1, pos_j + jjj + 1)
                reward_param = np.random.beta(pos_i + 1, pos_j + 1)
                reward = np.random.binomial(1, reward_param)
                if t > T * percent / 100:
                    total_reward += reward
                stay = np.random.binomial(1, wsp)  # see if it will stay

                if stay == 1:  # if the assigned worker is staying
                    if pos_i + pos_j < (state - 1):
                        queue[pos_i + 1][pos_j] += reward
                        queue[pos_i][pos_j + 1] += (1 - reward)
                    else:
                        queue[pos_i][pos_j] += 1

        for (i, j) in product(range(state), range(state)):
            queue[i][j] = min(queue[i][j], bigK)
            queue_mid[i][j] += queue[i][j]
            # mov_avg += 1

        # if t > T * percent / 100:
        #     # last_queues[counterr][0] = counterr + 1
        #     # index = 0
        #     for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
        #         # last_queues[counterr][index + 1] = queue[i][j]
        #         # index += 1
        #         pricesHere[i][j] += queue[i][j]
        #     # counterr += 1

        if int((t + 1) / 10) == ((t + 1) / 10):
            track_mass[counter_conv][0] = counter_conv + 1
            track_queues[counter_conv][0] = counter_conv + 1
            track_queues_cum[counter_conv][0] = counter_conv + 1
            index = 0
            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                track_mass[counter_conv][index + 1] = track_assign[i][j] / (t + 1)
                track_queues[counter_conv][index + 1] = queue[i][j]  # queue_mid[i][j] / (t + 1)
                track_queues_cum[counter_conv][index + 1] = queue_mid[i][j] / (t + 1)
                index += 1
            counter_conv += 1
    total_reward = total_reward / (T * (1 - percent / 100))
    pricesHere = pricesHere / (T * (1 - percent / 100))
    # print(track_queues[-1, :])
    return track_mass, total_reward, track_queues



state = 3
workerarriveprob = 0.1
workerstayprob = 0.95
jobarriveprob = 0.5

objVals = np.zeros((state, state))
objVals[0][0] = 0.5 + 0.11
objVals[0][1] = 0.4
objVals[0][2] = 0.3
# objVals[0][3] = 0.2
objVals[1][0] = 0.7
objVals[1][1] = 0.6
# objVals[1][2] = 0.5
objVals[2][0] = 0.8
# objVals[2][1] = 0.7
# objVals[3][0] = 0.9

TT = int(1e6)

track_mass, total_reward, track_queues = succfailSim(state, TT, workerarriveprob, jobarriveprob, workerstayprob,
                                                     1000, objVals, 2 * objVals[state - 1][0], 80)

df_massTree = pd.DataFrame(track_mass, columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                                ((i, j) for (i, j) in
                                                                 product(range(state), range(state)) if
                                                                 i + j <= (state - 1))]], dtype=float)

df_massTree.to_csv("massesOverTime.csv", index=False)

# plot_mass = ggplot(df_massTree) + aes(x='Time', y=['State(0,0)', 'State(0,1)', 'State(0,2)',
#                                                    'State(1,0)', 'State(1,1)', 'State(2,0)']) + geom_point(size=0.005)


# plot_mass = ggplot(aes(x="Time"), data=df_massTree) + \
#             geom_line(aes(y='State(0,0)'), color='red') + \
#             geom_line(aes(y='State(0,1)'), color='yellow') + \
#             geom_line(aes(y='State(0,2)'), color='green') + \
#             geom_line(aes(y='State(1,0)'), color='cyan') + \
#             geom_line(aes(y='State(1,1)'), color='blue') + \
#             geom_line(aes(y='State(2,0)'), color='purple')
#
# ggsave(plot_mass, filename="massConvergence.png")
raise SystemExit(0)

########################
## simulation performance is too bad, opt is 0.3575, fixed point is 0.3420, simulation is 0.2735
# state = 3
# timeHorz = 250000  # number of time periods
# bigK = 1e3
#
# rewardMultipliers = np.zeros((state, state))
# rewardMultipliers[2][0] = 0.89690722
# rewardMultipliers[1][0] = 0.81443299
# rewardMultipliers[1][1] = 0.74226804
# rewardMultipliers[0][0] = 0.70103093
# rewardMultipliers[0][1] = 0.12371134
# rewardMultipliers[0][2] = 0.10309278
# cC = 2 * rewardMultipliers[state - 1][0]
# massTree, empRewardTree, queuesTree = succfailSim(state, timeHorz, 0.15976464, 0.41081721,
#                                                   0.95818374, bigK, rewardMultipliers, cC, 80)
#
# print(empRewardTree)
# raise SystemExit(0)
########################

def simModulePriceDev(state, sims):
    keepRewards = np.zeros((sims, 6))
    index = 0
    for ss in range(sims):
        print("Iter ", ss + 1)
        workerarriveprob = random.uniform(0.1, 0.3)
        jobarriveprob = random.uniform(0.5, 0.7)
        wsp = random.uniform(0.85, 0.98)
        rewardMultips = np.array(sorted(random.sample(range(10, 100), int(state * (state + 1) / 2)), reverse=True))
        rewardMultips = rewardMultips / (max(rewardMultips) + min(rewardMultips))
        rewardMultipliers = np.zeros((state, state))
        # need a loop here
        ind = 0
        for i in range(state - 1, -1, -1):
            for j in range(state - i):
                if j <= i:
                    rewardMultipliers[i][j] = rewardMultips[ind]
                    ind += 1
        dist = random.sample(range(0, int(state * (state + 1) / 2) - ind), int(state * (state + 1) / 2) - ind)
        loc = 0
        for i in range(state - 1, -1, -1):
            for j in range(state - i):
                if j > i:
                    rewardMultipliers[i][j] = rewardMultips[ind + dist[loc]]
                    loc += 1
        # need a loop here
        print(rewardMultipliers)
        print("lambda is ", workerarriveprob, ", mu is ", jobarriveprob, " beta is ", wsp, "\n")

        # call the optimization model first
        xmatrix_opt, objval_opt, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob,
                                                      np.zeros((state, state)), False, rewardMultipliers)
        # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
        if jobcon - jobarriveprob > -1e-8:
            # then the fixed point
            fixedPointTree, fixedPoint_objval = succfailFixedPoint(state, wsp, workerarriveprob,
                                                                   jobarriveprob, rewardMultipliers)

            if fixedPoint_objval > 0:
                # then the dual prices corresponding to the fixed point
                succfailFixedPointDual = succfailDualUseFixedPoint(state, wsp, workerarriveprob, jobarriveprob,
                                                                   fixedPointTree, rewardMultipliers)
                maxDualPrice = succfailFixedPointDual.max()
                # then the simulation
                timeHorz = 500000  # number of time periods
                bigK = 1e3
                cC = 2 * rewardMultipliers[state - 1][0]
                percent = 80  # Last portion
                print("Now the simulation in iter", ss + 1, "\n")

                massTree, empRewardTree, queuesTree = succfailSim(state, timeHorz, workerarriveprob,
                                                                  jobarriveprob, wsp, bigK,
                                                                  rewardMultipliers, cC, percent)
                df_qsTree = pd.DataFrame(queuesTree,
                                         columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                                    ((i, j) for (i, j) in
                                                                     product(range(state), range(state)) if
                                                                     i + j <= (state - 1))]], dtype=int)

                pricesFromSim = np.zeros((state, state))
                for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                    name = 'State(' + str(i) + ',' + str(j) + ')'
                    namep_just = 'Price(' + str(i) + ',' + str(j) + ')'

                    # local price #
                    df_qsTree[namep_just] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)

                for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                    name = 'Price(' + str(i) + ',' + str(j) + ')'
                    mid = df_qsTree[name]
                    midd = mid[mid.columns[0]].values
                    pricesFromSim[i][j] = statistics.mean(midd[-int((timeHorz * (1 - percent / 100)) / 10):])
                print(pricesFromSim)
                print(succfailFixedPointDual)
                for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                    if succfailFixedPointDual[i][j] == 0:
                        pricesFromSim[i][j] = 0
                diffprices = np.abs(succfailFixedPointDual - pricesFromSim)
                maxdiff = diffprices.max()
                pos_i = np.unravel_index(np.argmax(diffprices, axis=None), diffprices.shape)[0]
                pos_j = np.unravel_index(np.argmax(diffprices, axis=None), diffprices.shape)[1]
                keepRewards[index][0] = objval_opt
                keepRewards[index][1] = fixedPoint_objval
                keepRewards[index][2] = fixedPoint_objval / objval_opt
                keepRewards[index][3] = maxdiff / maxDualPrice
                keepRewards[index][4] = pos_i
                keepRewards[index][5] = pos_j
                index += 1
    print(keepRewards)


def simModule(state, sims):
    ####simulation module####
    keepRewards = np.zeros((sims, 3))
    index = 0
    for ss in range(sims):
        print("Iter ", ss + 1)
        workerarriveprob = random.uniform(0.1, 0.3)
        jobarriveprob = random.uniform(0.5, 0.7)
        wsp = random.uniform(0.85, 0.98)
        rewardMultips = np.array(sorted(random.sample(range(10, 100), int(state * (state + 1) / 2)), reverse=True))
        rewardMultips = rewardMultips / (max(rewardMultips) + min(rewardMultips))
        rewardMultipliers = np.zeros((state, state))

        # need a loop here
        ind = 0
        for i in range(state - 1, -1, -1):
            for j in range(state - i):
                if j <= i:
                    rewardMultipliers[i][j] = rewardMultips[ind]
                    ind += 1
        dist = random.sample(range(0, int(state * (state + 1) / 2) - ind), int(state * (state + 1) / 2) - ind)
        loc = 0
        for i in range(state - 1, -1, -1):
            for j in range(state - i):
                if j > i:
                    rewardMultipliers[i][j] = rewardMultips[ind + dist[loc]]
                    loc += 1
        # need a loop here

        print(rewardMultipliers)
        print("lambda is ", workerarriveprob, ", mu is ", jobarriveprob, " beta is ", wsp, "\n")

        # call the optimization model first
        xmatrix_opt, objval_opt, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob,
                                                      np.zeros((state, state)), False, rewardMultipliers)
        # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
        if jobcon - jobarriveprob > -1e-8:
            # then the fixed point
            fixedPointTree, fixedPoint_objval = succfailFixedPoint(state, wsp, workerarriveprob,
                                                                   jobarriveprob, rewardMultipliers)
            # then the simulation
            # timeHorz = 200000  # number of time periods
            # bigK = 1e3
            # cC = 2 * rewardMultipliers[state - 1][0]
            # percent = 80  # Last portion
            # print("Now the simulation in iter", ss + 1, "\n")
            # treerewards = 0
            # numsim = 5
            # for jkks in range(numsim):
            #     print("Simulation number ", jkks + 1)
            #     massTree, empRewardTree, queuesTree = succfailSim(state, timeHorz, workerarriveprob,
            #                                                                        jobarriveprob, wsp, bigK,
            #                                                                        rewardMultipliers, cC, percent)
            #     treerewards += empRewardTree
            keepRewards[index][0] = objval_opt
            keepRewards[index][1] = fixedPoint_objval
            keepRewards[index][2] = fixedPoint_objval / objval_opt
            index += 1
    print(keepRewards)
    # np.savetxt("keepRewards.csv", keepRewards[0:index], delimiter=",")


def getRs(rewardMulti, state):
    rewardMulti = rewardMulti / (max(rewardMulti) + min(rewardMulti))
    rs = np.zeros((state, state))
    rs[4][0] = rewardMulti[0]
    rs[3][0], rs[3][1] = rewardMulti[1], rewardMulti[2]
    rs[2][0], rs[2][1], rs[2][2] = rewardMulti[3], rewardMulti[4], rewardMulti[5]
    rs[1][0], rs[1][1] = rewardMulti[6], rewardMulti[7]
    rs[0][0] = rewardMulti[8]
    rs[1][2], rs[1][3] = rewardMulti[9], rewardMulti[11]
    rs[0][1], rs[0][2], rs[0][3], rs[0][4] = rewardMulti[10], rewardMulti[12], rewardMulti[13], rewardMulti[14]
    return rs


def simModuleDemandVarying(instance):
    state = 5
    workerarriveprob = 1
    wsp = 0.75
    alphas = [(i + 1) / 10 for i in range(10)]
    keepResults = np.zeros((10, 6))
    for ss in range(len(alphas)):
        jobarriveprob = workerarriveprob / (1 - wsp) * alphas[ss]  # alphas[len(alphas) - ss - 1]

        sims = 0
        keepMid = np.zeros((instance, 7))
        while sims < instance:
            rewardMulti = np.array(sorted(random.sample(range(10, 100), 15), reverse=True))
            rs = getRs(rewardMulti, state)
            # call the optimization model first
            xmatrix_opt, objval_opt, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob,
                                                          np.zeros((state, state)), False, rs)
            # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
            if jobcon - jobarriveprob > -1e-8:
                fixedPointTree, fixedPoint_objval = succfailFixedPoint(state, wsp, workerarriveprob, jobarriveprob, rs)
                if fixedPoint_objval > 0:
                    # then the dual prices corresponding to the fixed point
                    FPDual = succfailDualUseFixedPoint(state, wsp, workerarriveprob, jobarriveprob, fixedPointTree, rs)
                    maxDualPrice = 0
                    for (i, j) in ((i, j) for (i, j) in product(range(state), range(state))
                                   if (i + j <= (state - 1)) & (fixedPointTree[i][j] > 0)
                                      & (FPDual[i][j] > maxDualPrice)):
                        maxDualPrice = FPDual[i][j]
                    # then the simulation
                    timeHorz = 200000  # number of time periods
                    bigK = 1e2 if jobarriveprob <= 0.5 else 1e3
                    cC = 2 * rs[state - 1][0]
                    percent = 80  # Last portion
                    print("\nsim ", sims, " with mu ", jobarriveprob, " and w BigK ", bigK, "\n")
                    mass, empRew, queuesTree = succfailSim(state, timeHorz, workerarriveprob, jobarriveprob, wsp,
                                                           bigK, rs, cC, percent)
                    pricesFromSim = np.zeros((state, state))
                    index = 1
                    for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                        mid = queuesTree[:, index]
                        midd = statistics.mean(mid[-int((timeHorz * (1 - percent / 100)) / 1):])
                        pricesFromSim[i][j] = cC * (bigK - min(bigK, midd)) / bigK
                        index += 1
                    print()
                    print(pricesFromSim)
                    print(FPDual)
                    for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if
                                   i + j <= (state - 1)):
                        if FPDual[i][j] == 0:
                            pricesFromSim[i][j] = 0
                    print(pricesFromSim)
                    diffprices = np.abs(FPDual - pricesFromSim)
                    maxdiff = diffprices.max()
                    keepMid[sims][0] = objval_opt
                    keepMid[sims][1] = fixedPoint_objval
                    keepMid[sims][2] = empRew
                    keepMid[sims][3] = fixedPoint_objval / objval_opt
                    keepMid[sims][4] = empRew / objval_opt
                    keepMid[sims][5] = maxdiff / maxDualPrice
                    print("\nPrice deviation", maxdiff / maxDualPrice)
                    sims += 1
        for jj in range(6):
            keepResults[ss][jj] = statistics.mean(keepMid[:, jj])
        np.savetxt("EC-sims_keepResults.csv", keepResults, delimiter=",")
    print()
    print(keepResults)


def simModulePriorsChange(state, sims):
    workerarriveprob = 0.5
    wsp = 0.95
    alphas = [(i + 1) / (1 * 10) for i in range(10)]
    keepResults = np.zeros((len(alphas), 3))
    simstart, simend = 0, 0
    for ss in range(len(alphas)):
        jobarriveprob = 10 * alphas[ss]
        # jobarriveprob = workerarriveprob / (1 - wsp) * alphas[ss]  # alphas[len(alphas) - ss - 1]
        # jobarriveprob = jobarriveprob if jobarriveprob <= 1 else 1

        # keepMid = np.zeros((int(state * (state + 1) / 2), 3))
        priorstuff = 5
        keepMid = np.zeros((priorstuff, 3))
        ind = 0
        # for (ii, jj) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
        #     zeroprior_s = ii
        #     zeroprior_f = jj
        for jj in range(priorstuff):
            zeroprior_f = jj
            # get the objective, each element keeps the ACTUAL success probability
            objective = np.zeros((state, state))
            for (iii, jjj) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                objective[iii][jjj] = (iii + 1) / (iii + jjj + zeroprior_f + 2)
            print(objective)
            # call the optimization model first
            objval_opt, jobcon = succfailOptPriors(state, wsp, workerarriveprob, jobarriveprob, objective)
            # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
            if jobcon - jobarriveprob > -1e-8:
                FP_objval = 0
                FPTree, FP_objval, solnchange, both = succfailFixedPointPriors(state, wsp, workerarriveprob,
                                                                         jobarriveprob, objective)
                print("and the optimal value is ", objval_opt, "\n and LME search found a new solution ", solnchange,
                      " times and in the first try ", both + 1, " LME objs give a solution")
                if FP_objval > 0:
                    # then the simulation
                    simsRewAvg = 0
                    for sss in range(sims):
                        timeHorz = 2500 #if jobarriveprob > 1 else 500000 # number of time periods
                        bigK = 1e2  # if jobarriveprob <= 1.91 else 1e3
                        cC = 2 * objective.max()
                        percent = 80  # Last portion
                        print(keepMid)
                        print("\nsim ", sss, " of ", ind, " instance with mu ", jobarriveprob, # + 0.1,
                              " and w BigK ", bigK,
                              "\n with previous sim taking ", simend - simstart, " seconds")

                        simstart = time.time()
                        # mass, empRew, queuesTree = succfailSim(state, timeHorz, workerarriveprob, jobarriveprob, wsp,
                        #                                        bigK, objective, cC, percent, zeroprior_s, zeroprior_f)
                        mass, empRew, queuesTree = succfailSim(state, timeHorz, workerarriveprob, jobarriveprob, wsp,
                                                               bigK, objective, cC, percent, 0, zeroprior_f)
                        simend = time.time()
                        simsRewAvg += empRew
                        print("This runs reward is ", empRew, " and the optimal was ", objval_opt, ", a ratio of ",
                              empRew / objval_opt)
                    simsRewAvg = simsRewAvg / sims
            keepMid[ind][0] = FP_objval / objval_opt
            keepMid[ind][1] = simsRewAvg / objval_opt
            keepMid[ind][2] = simsRewAvg / FP_objval
            print(keepMid)
            ind += 1
        filename = 'Prior_alpha' + str(ss) + 'mu' + str(jobarriveprob) + '_newLME.csv'
        np.savetxt(filename, keepMid, delimiter=",")
        keepResults[ss][0] = statistics.mean(keepMid[:, 0])
        keepResults[ss][1] = statistics.mean(keepMid[:, 1])
        keepResults[ss][2] = statistics.mean(keepMid[:, 2])
        midtime = time.time()
        print("It has been ", midtime - start, " seconds so far, from the start that is")
        np.savetxt("EC-NewSims_keepResults_5priors.csv", keepResults, delimiter=",")
        print()
        print(keepResults)


####simulation module####


start = time.time()
# n = 10
# numsim = 1

# simModulePriorsChange(n, numsim)
# simModuleDemandVarying(numsim)
# simModule(n, numsim)
# simModulePriceDev(n, numsim)


# end = time.time()
# print("It took ", end - start, " seconds for the whole thing")
# raise SystemExit(0)

state = 3  # there are this many possible advances
workerarriveprob = 0.2  # at each time period, a worker arrives with this probability
jobarriveprob = 0.8  # at each time period, a job arrives with this probability
wsp = 0.99  # workerstayprobability: probability of worker staying in the system after completing a job

rewardMultipliers = [[(i + 1) / (i + j + 2) if (i + j) < state else 0 for j in range(state)] for i in range(state)]

rewardMultipliers[1][0] = 0.999
rewardMultipliers[2][0] = 0.9991
rewardMultipliers[1][1] = 0.00011
rewardMultipliers[0][0] = 0.0001
rewardMultipliers[0][1] = rewardMultipliers[0][0] / 2
rewardMultipliers[0][2] = rewardMultipliers[0][0] / 3

print(rewardMultipliers)

succfailDual(state, wsp, workerarriveprob, jobarriveprob, rewardMultipliers)

# get the fixed point
fixedPointTree, objval_irrev = succfailFixedPoint(state, wsp, workerarriveprob, jobarriveprob, rewardMultipliers)
#
print("\nfixed point is")
print(fixedPointTree)
print()

# # primal optimization problem
print("CALLING THE OPTIMIZATION MODEL")
xmatrix = np.zeros((state, state))
xmatrix_opt, obj_val_opt, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob, xmatrix, False,
                                               rewardMultipliers)
# raise SystemExit(0)
print()
xmatrix_check, obj_v, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob, xmatrix_opt, True,
                                           rewardMultipliers)
if np.abs(obj_v - obj_val_opt) > 1e-8:
    # warnings.warn("NOT AN LME!")
    print("NOT AN LME!\n")

xmatrix[0][0] = 0.047619047619047734  # 0.066740823136818742  #
xmatrix[1][0] = 0.022619047619047643  # 0.031701890989988893  #
xmatrix[1][1] = 0.143253968235847184  # 0  #
xmatrix[2][0] = 0.286507936526057438  # 0.401557285873192393  #
# xmatrix = xmatrix_opt
useprevsoln = True
max_abs_diff = 1
precision = 1e-12
iter = 0
maxiter = 2000
while (max_abs_diff > precision) & (iter < maxiter):
    prev_xmatrix = deepcopy(xmatrix)
    xmatrix, obj_val, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob, xmatrix, useprevsoln,
                                           rewardMultipliers)
    max_abs_diff = np.max(np.abs(prev_xmatrix - xmatrix))
    print(f"Maximum absolute difference is {max_abs_diff:.10f}")
    print('iter ', iter)
    iter += 1
print("Objective value is", obj_val_opt, ", Optimal soln is")
print(xmatrix_opt)
print("\nthe LME objective ratio to the optimal value is %", obj_val / obj_val_opt * 100)

print(rewardMultipliers)
raise SystemExit(0)

# getting the dual prices using the solutions
print(xmatrix)
print("using a fixed point for the rhs")
succfailFixedPointDual = succfailDualUseFixedPoint(state, wsp, workerarriveprob, jobarriveprob,
                                                   xmatrix, rewardMultipliers)
print()
print("using the optimal solution for the rhs")
print(xmatrix_opt)
succfailFixedPointDual_wOPT = succfailDualUseFixedPoint(state, wsp, workerarriveprob, jobarriveprob,
                                                        xmatrix_opt, rewardMultipliers)

# raise SystemExit(0)

# # simulation
timeHorz = 500000  # number of time periods
bigK = 1e3
cC = 2 * rewardMultipliers[state - 1][0]
percent = 80
massTree, empRewardTree, queuesTree, queuesTreeCumul = succfailSim(state, timeHorz, workerarriveprob, jobarriveprob,
                                                                   wsp, bigK, rewardMultipliers, cC, percent)
# dataframes
df_massTree = pd.DataFrame(massTree, columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                                ((i, j) for (i, j) in
                                                                 product(range(state), range(state)) if
                                                                 i + j <= (state - 1))]], dtype=float)
df_qsTree = pd.DataFrame(queuesTree, columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                                ((i, j) for (i, j) in
                                                                 product(range(state), range(state)) if
                                                                 i + j <= (state - 1))]], dtype=int)
df_cumulQsTree = pd.DataFrame(queuesTreeCumul, columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')'
                                                                          for (i, j)
                                                                          in ((i, j) for (i, j) in
                                                                              product(range(state), range(state)) if
                                                                              i + j <= (state - 1))]], dtype=int)

pd.set_option('display.max_columns', None)
print(df_massTree)

for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
    name = 'State(' + str(i) + ',' + str(j) + ')'
    namep = 'PriceEff(' + str(i) + ',' + str(j) + ')'
    namep_just = 'Price(' + str(i) + ',' + str(j) + ')'

    # local price #
    df_qsTree[namep_just] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    df_cumulQsTree[namep_just] = df_cumulQsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    # local price #

    # optimal price #
    # name_s = 'State(' + str(i + 1) + ',' + str(j) + ')'
    # name_f = 'State(' + str(i) + ',' + str(j + 1) + ')'
    # if i + j < (state - 1):
    #     if (i + j == 0) or (i * j >= 1):
    #         df_qsTree[namep] = df_qsTree.apply(
    #             lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
    #                             wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, x[name_s])) / bigK -
    #                             wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, x[name_f])) / bigK), axis=1)
    #     elif i == 0:
    #         df_qsTree[namep] = df_qsTree.apply(
    #             lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
    #                             wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, x[name_f])) / bigK), axis=1)
    #     elif j == 0:
    #         df_qsTree[namep] = df_qsTree.apply(
    #             lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
    #                             wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, x[name_s])) / bigK), axis=1)
    # else:
    #     df_qsTree[namep] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    # df_qsTree[namep_just] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    # optimal price #

# print(df_laterQsTree)
df_qsTree.to_csv("qs_and_localprices.csv", index=False)
df_cumulQsTree.to_csv("qs_and_localprices_cumulative.csv", index=False)
raise SystemExit(0)

for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
    name = 'State(' + str(i) + ',' + str(j) + ')'
    plot = ggplot(df_massTree) + aes(x='Time', y=name) + geom_line() \
           + geom_hline(yintercept=fixedPointTree[i][j], color="red")
    # + geom_hline(yintercept=xmatrix_opt[i][j], color="red")
    ggsave(plot, filename=name)
    plot_queue = ggplot(df_qsTree) + aes(x='Time', y=name) + geom_point(size=0.005)
    nameq = 'Queue(' + str(i) + ',' + str(j) + ')'
    ggsave(plot_queue, filename=nameq)
    # namep = 'PriceEff(' + str(i) + ',' + str(j) + ')'
    # plot_price = ggplot(df_qsTree) + aes(x='Time', y=namep) + geom_point(size=0.005)
    #              # + geom_hline(yintercept=succfailFixedPointDual[i][j], color="red")
    # ggsave(plot_price, filename=namep)
    namep_just = 'Price(' + str(i) + ',' + str(j) + ')'
    plot_pricej = ggplot(df_qsTree) + aes(x='Time', y=namep_just) + geom_point(size=0.005) \
                  + geom_hline(yintercept=succfailFixedPointDual[i][j], color="red")
    ggsave(plot_pricej, filename=namep_just)

fp_obj_val = xsum(fixedPointTree[i][j] * rewardMultipliers[i][j] for j in range(state) for i in range(state))
print('Empirical reward is ', empRewardTree / timeHorz, ' and the reward due to the fixed point is ', fp_obj_val)
