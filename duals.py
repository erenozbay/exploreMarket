from itertools import product
from mip import *
import numpy as np


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
    print("alpha " + str(alpha.x) + "; and the solution")
    for (i, j) in product(range(n), range(n)):
        soln[i][j] = g[i][j].x
    print(soln)
    return soln


# dual of the tree model using a fixed point for the right hand-side
def succfailDualUseFixedPoint(n, beta, lambd, mu, fixedPoint, objective):
    x = fixedPoint
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
            if j == n - 1:
                rhs[j][i] = beta * (j / (i + j + 1)) * x[j - 1][i] + beta * x[j][i]
            else:
                rhs[j][i] = beta * (j / (i + j + 1)) * x[j - 1][i]
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if i == n - 1:
                rhs[j][i] = beta * (i / (i + j + 1)) * x[j][i - 1] + beta * x[j][i]
            else:
                rhs[j][i] = beta * (i / (i + j + 1)) * x[j][i - 1]

    m = Model()
    g = [[m.add_var(name='gamma({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    alpha = m.add_var(lb=0)

    for (i, j) in product(range(n), range(n)):
        m += g[i][j] + alpha >= objective[i][j]
    m.objective = minimize(xsum(rhs[i][j] * g[i][j] for i in range(n) for j in range(n)) + alpha * mu)

    m.optimize()

    print('Objective value is ', m.objective_value)
    soln = np.zeros((n, n))
    print("alpha " + str(alpha.x) + "; and the solution")
    for (i, j) in product(range(n), range(n)):
        soln[i][j] = g[i][j].x
    print(soln)
    return soln
