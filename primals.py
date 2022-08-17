# this is for the tree model
from itertools import product
from mip import *
import numpy as np

# primal model
# def succfailOptPriors(n, beta, lambd, mu, objective):
#     m = Model()
#     x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
#           for i in range(n)] for j in range(n)]
#     ys = [[m.add_var(name='slack({},{})'.format(j + 1, i + 1), lb=0)
#            for i in range(n)] for j in range(n)]
#     m.objective = maximize(xsum(objective[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1))
#
#     for (j, i) in product(range(n), range(n)):
#         if (i > 0) & (j > 0) & (j + i <= n - 1):
#             if j + i == n - 1:
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - \
#                      beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - \
#                      beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
#             else:
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - \
#                      beta * (1 - objective[j][i - 1]) * x[j][i - 1] <= 0
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - \
#                      beta * (1 - objective[j][i - 1]) * x[j][i - 1] == - ys[j][i]
#         elif (i == 0) & (j != 0) & (j + i <= n - 1):
#             if j + i == n - 1:
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - beta * x[j][i] <= 0
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] - beta * x[j][i] == - ys[j][i]
#             else:
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] <= 0
#                 m += x[j][i] - beta * objective[j - 1][i] * x[j - 1][i] == - ys[j][i]
#         elif (i != 0) & (j == 0) & (j + i <= n - 1):
#             if j + i == n - 1:
#                 m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
#                 m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
#             else:
#                 m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] <= 0
#                 m += x[j][i] - beta * (1 - objective[j][i - 1]) * x[j][i - 1] == - ys[j][i]
#
#     m += xsum(x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) <= mu
#
#     m += x[0][0] <= lambd
#     m.optimize()
#     mass = 0
#     for (j, i) in product(range(n), range(n)):
#         mass += x[j][i].x
#
#     if m.optimize() != OptimizationStatus.INFEASIBLE:
#         print("The job constraint is %.10f" % xsum(x[j][i].x for j in range(n) for i in range(n)))
#         print("Objective: ", m.objective_value)
#         for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
#             print("{:<2d} successes and {:<2d} failures have mass "
#                   "{:<.18f} and slack {:<.18f}".format(i + 1, j + 1, x[i][j].x, ys[i][j].x))
#
#     return m.objective_value, mass


def succfailOpt(n, beta, lambd, mu, prevSoln, usePrevSoln, objective):
    m = Model()
    x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    ys = [[m.add_var(name='slack({},{})'.format(j + 1, i + 1), lb=0)
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
    mass = 0
    for (j, i) in product(range(n), range(n)):
        newSoln[j][i] = x[j][i].x
        mass += x[j][i].x

    if m.optimize() != OptimizationStatus.INFEASIBLE:
        print("The job constraint is %.10f" % xsum(x[j][i].x for j in range(n) for i in range(n)))
        print("Objective: ", m.objective_value)
        for (i, j) in ((i, j) for (i, j) in product(range(n), range(n)) if i + j <= (n - 1)):
            print("{:<2d} successes and {:<2d} failures have mass "
                  "{:<.18f} and slack {:<.18f}".format(i + 1, j + 1, x[i][j].x, ys[i][j].x))

    return newSoln, m.objective_value, mass

# bounds and slacks will be matrices, in the slacks matrix, the indices with 1 will be forced to have zero slacks
# def succfailOptFixedPoint(n, beta, lambd, mu, bounds, slacks, objMult, whichobj):
#     m = Model()
#     x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
#           for i in range(n)] for j in range(n)]
#     y = m.add_var()
#     if whichobj == 1:
#         print("regular objective")
#         m.objective = maximize(y)
#     else:
#         print("other objective")
#         m.objective = minimize(y)
#         m += y >= 0
#
#     if bounds.sum() > mu:
#         for (i, j) in ((i, j) for j in range(n) for i in range(n) if (j + i <= n - 1) & (slacks[i][j] == 1)):
#             if (i + j) == 0:
#                 m += x[0][0] >= lambd
#                 # m += y[i][j] <= lambd - x[0][0]
#             elif (i == 0) & (j != 0):
#                 if j + i == n - 1:
#                     m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] >= 0
#                     # m += y[i][j] >= beta * (j / (i + j + 1)) * x[i][j - 1] + beta * x[i][j] - x[i][j]
#                 else:
#                     m += x[i][j] - beta * (j / (i + j + 1)) * x[i][j - 1] >= 0
#                     # m += y[i][j] >= beta * (j / (i + j + 1)) * x[i][j - 1] - x[i][j]
#             elif (i != 0) & (j == 0):
#                 if j + i == n - 1:
#                     m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * x[i][j] >= 0
#                     # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] + beta * x[i][j] - x[i][j]
#                 else:
#                     m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] >= 0
#                     # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] - x[i][j]
#             else:
#                 if j + i == n - 1:
#                     m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - \
#                          beta * (j / (i + j + 1)) * x[i][j - 1] - beta * x[i][j] >= 0
#                     # m += y[i][j] >= beta * (i / (i + j + 1)) * x[i - 1][j] + \
#                     #      beta * (j / (i + j + 1)) * x[i][j - 1] + beta * x[i][j] - x[i][j]
#                 else:
#                     m += x[i][j] - beta * (i / (i + j + 1)) * x[i - 1][j] - beta * (j / (i + j + 1)) * x[i][
#                         j - 1] >= 0
#
#     for (j, i) in product(range(n), range(n)):
#         m += x[j][i] <= bounds[j][i]
#         if (i > 0) & (j > 0) & (j + i <= n - 1):
#             if j + i == n - 1:
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
#                      beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= 0
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
#                      beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= -y
#             else:
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= 0
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - \
#                      beta * (i / (i + j + 1)) * x[j][i - 1] <= -y
#         elif (i == 0) & (j != 0) & (j + i <= n - 1):
#             if j + i == n - 1:
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * x[j][i] <= 0
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] - beta * x[j][i] <= -y
#             else:
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] <= 0
#                 m += x[j][i] - beta * (j / (i + j + 1)) * x[j - 1][i] <= -y
#         elif (i != 0) & (j == 0) & (j + i <= n - 1):
#             if j + i == n - 1:
#                 m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= 0
#                 m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] - beta * x[j][i] <= -y
#             else:
#                 m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= 0
#                 m += x[j][i] - beta * (i / (i + j + 1)) * x[j][i - 1] <= -y
#
#     m += xsum(x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1) == mu
#
#     m += x[0][0] <= lambd
#     m.optimize()
#
#     soln_sub = np.zeros((n, n))
#     obj = 0
#     for (j, i) in product(range(n), range(n)):
#         soln_sub[j][i] = x[j][i].x
#         obj += objMult[j][i] * soln_sub[j][i]
#     print("Objective is ", obj, ", and y is ", y.x)
#     return soln_sub, m.optimize()


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
    elif whichobj == 0:
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
