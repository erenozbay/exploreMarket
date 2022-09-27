# from itertools import product
from mip import *
import numpy as np
import pandas as pd
from copy import deepcopy
from duals import *


#### THESE SHOULD USE THE TRANSITIONS, DUALS ARE NOT NECESSARILY FOLLOWING THE REGULAR TREE TRANSITIONS
# dual of the tree model using a fixed point
def DualUseFixedPoint(n, beta, lambd, mu, x, objective):
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
    print("alpha " + str(alpha.x) + "; and the solution")
    for (i, j) in product(range(n), range(n)):
        soln[i][j] = g[i][j].x
    print(soln)
    return soln


def OptPriors(n, beta, lambd, mu, objective, transitions):
    m = Model()
    x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    ys = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
           for i in range(n)] for j in range(n)]
    m.objective = maximize(xsum(objective[j][i] * x[j][i] for j in range(n) for i in range(n) if j + i <= n - 1))

    for (j, i) in product(range(n), range(n)):
        if (i > 0) & (j > 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
            else:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - beta * (1 - transitions[j][i - 1]) * x[j][
                    i - 1] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - transitions[j][i - 1]) * x[j][i - 1] == - ys[j][i]
        elif (i == 0) & (j != 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - beta * x[j][i] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - beta * x[j][i] == - ys[j][i]
            else:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] == - ys[j][i]
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] == - ys[j][i]
            else:
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] <= 0
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] == - ys[j][i]

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
    soln_sub = np.zeros((n, n))
    for (j, i) in product(range(n), range(n)):
        soln_sub[j][i] = x[j][i].x
    return m.objective_value, jobcon, soln_sub


def OptFPPriors(n, beta, lambd, mu, bounds, slacks, objMult, whichobj, transitions):
    m = Model()
    x = [[m.add_var(name='x({},{})'.format(j + 1, i + 1), lb=0)
          for i in range(n)] for j in range(n)]
    y = m.add_var()
    if whichobj == 1:
        print("regular objective")
        m.objective = maximize(y)
        # m.objective = maximize(xsum(objMult[j][i] * x[j][i]
        #                             for j in range(n) for i in range(n) if j + i <= n - 1))
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
            elif (i == 0) & (j != 0):
                if j + i == n - 1:
                    m += x[i][j] - beta * (1 - transitions[i][j - 1]) * x[i][j - 1] - beta * x[i][j] >= 0
                else:
                    m += x[i][j] - beta * (1 - transitions[i][j - 1]) * x[i][j - 1] >= 0
            elif (i != 0) & (j == 0):
                if j + i == n - 1:
                    m += x[i][j] - beta * transitions[i - 1][j] * x[i - 1][j] - beta * x[i][j] >= 0
                else:
                    m += x[i][j] - beta * transitions[i - 1][j] * x[i - 1][j] >= 0
            else:
                if j + i == n - 1:
                    m += x[i][j] - beta * transitions[i - 1][j] * x[i - 1][j] - \
                         beta * (1 - transitions[i][j - 1]) * x[i][j - 1] - beta * x[i][j] >= 0
                else:
                    m += x[i][j] - beta * transitions[i - 1][j] * x[i - 1][j] - \
                         beta * (1 - transitions[i][j - 1]) * x[i][j - 1] >= 0

    for (j, i) in product(range(n), range(n)):
        m += x[j][i] <= bounds[j][i]
        if (i > 0) & (j > 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - transitions[j][i - 1]) * x[j][i - 1] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - \
                     beta * (1 - transitions[j][i - 1]) * x[j][i - 1] <= -y
        elif (i == 0) & (j != 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - beta * x[j][i] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] <= 0
                m += x[j][i] - beta * transitions[j - 1][i] * x[j - 1][i] <= -y
        elif (i != 0) & (j == 0) & (j + i <= n - 1):
            if j + i == n - 1:
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= 0
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] - beta * x[j][i] <= -y
            else:
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] <= 0
                m += x[j][i] - beta * (1 - transitions[j][i - 1]) * x[j][i - 1] <= -y

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


def FixedPointPriors(n, beta, lambd, mu, objective, transitions):
    # print out the fixed point whenever you find it
    slacks, bounds = np.zeros((n, n)), np.zeros((n, n))
    # (0, 0) should be positive, hence the slacks for all states strictly better than (0, 0) should be zero
    bounds[0][0] = mu
    minrew = 100
    for (i, j) in ((i, j) for j in range(n) for i in range(n) if (i + j <= n - 1)):
        if minrew >= objective[i][j]:
            minrew = objective[i][j]
        if objective[i][j] > objective[0][0]:
            slacks[i][j] = 1
            bounds[i][j] = mu
    print("\ncall model")
    print(bounds)

    objectivevalue = 0
    objectivevalue_b = 0
    finalobjval = 0
    changed_soln = 0
    both_solns = -1

    # argument 1 implies objective is maximize(a), argument 0 implies objective is minimize(a)
    a_soln, feasibility = OptFPPriors(n, beta, lambd, mu, bounds, slacks, objective, 1, transitions)
    if feasibility == OptimizationStatus.OPTIMAL:
        fp = a_soln
        print(fp)
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectivevalue += objective[i][j] * fp[i][j]

    a_soln_b, feasibility_b = OptFPPriors(n, beta, lambd, mu, bounds, slacks, objective, 0, transitions)
    if feasibility_b == OptimizationStatus.OPTIMAL:
        fp_b = a_soln_b
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectivevalue_b += objective[i][j] * fp_b[i][j]
        print(fp_b)

    if (objectivevalue > 0) & (objectivevalue_b > 0):
        keepsoln = fp if objectivevalue_b > objectivevalue else fp_b
        both_solns = 1
        finalobjval -= finalobjval
        finalobjval = min(objectivevalue, objectivevalue_b)
        if abs(objectivevalue - objectivevalue_b) > 1e-6:
            changed_soln += 1
        bestobjval = finalobjval
        returnsoln = keepsoln
    elif objectivevalue > 0:
        both_solns = 0
        keepsoln = fp
        finalobjval -= finalobjval
        finalobjval = objectivevalue
        bestobjval = finalobjval
        returnsoln = keepsoln
    elif objectivevalue_b > 0:
        both_solns = 0
        keepsoln = fp_b
        finalobjval -= finalobjval
        finalobjval = objectivevalue_b
        bestobjval = finalobjval
        returnsoln = keepsoln
    else:
        bestobjval = -10
        returnsoln = -1 * np.ones((n, n))

    success_prob = objective[0][0]  # the success probability of (0, 0)
    counter = 0
    while (success_prob >= minrew) and (counter < n * n * 0.6):
        counter += 1
        nextsuccess_prob = 0
        ind = -1
        possible_indices = np.zeros(((n + 3) // 2, 2))
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] <= success_prob)
                                                                      & (i + j <= n - 1))):
            if objective[i][j] >= success_prob:  # to see if there are multiple subsets to be tried, if so, look at
                # their combinations, if no, just add one by one
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
                        print(bounds)
                        objectivevalue = 0
                        firsttime = 0
                        for jk in range(2):
                            # print("call with multiple choices")
                            a_soln, feasibility = OptFPPriors(n, beta, lambd, mu, bounds,
                                                              slacks, objective, jk, transitions)
                            if (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 0):
                                keepsoln = a_soln
                                for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                                    objectivevalue += objective[i][j] * a_soln[i][j]
                                print(a_soln)
                                finalobjval -= finalobjval
                                finalobjval = objectivevalue
                                if finalobjval > bestobjval:
                                    returnsoln = keepsoln
                                    bestobjval = finalobjval
                                firsttime = 1
                            elif (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 1):
                                objectivevalue_b = 0
                                for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                                    objectivevalue_b += objective[i][j] * a_soln[i][j]
                                print(a_soln)
                                if objectivevalue_b < objectivevalue:
                                    finalobjval -= finalobjval
                                    finalobjval = objectivevalue_b
                                    keepsoln = a_soln
                                    if finalobjval > bestobjval:
                                        returnsoln = keepsoln
                                        bestobjval = finalobjval
                                    changed_soln += 1
                                    print(a_soln)
            else:
                # print(bounds)
                ii = int(possible_indices[ind][0])
                jj = int(possible_indices[ind][1])
                bounds[ii][jj] = mu
                print(bounds)
                objectivevalue = 0
                firsttime = 0
                for jk in range(2):
                    # print("call with single choice")
                    a_soln, feasibility = OptFPPriors(n, beta, lambd, mu, bounds,
                                                      slacks, objective, jk, transitions)
                    if (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 0):
                        keepsoln = a_soln
                        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                            objectivevalue += objective[i][j] * a_soln[i][j]
                        # print(a_soln)
                        finalobjval -= finalobjval
                        finalobjval = objectivevalue
                        if finalobjval > bestobjval:
                            returnsoln = keepsoln
                            bestobjval = finalobjval
                        firsttime = 1
                    elif (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 1):
                        objectivevalue_b = 0
                        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                            objectivevalue_b += objective[i][j] * a_soln[i][j]
                        if objectivevalue_b < objectivevalue:
                            finalobjval -= finalobjval
                            finalobjval = objectivevalue_b
                            keepsoln = a_soln
                        if finalobjval > bestobjval:
                            returnsoln = keepsoln
                            bestobjval = finalobjval
                            changed_soln += 1
                            # print(a_soln)

        success_prob = nextsuccess_prob
        # before moving on to the next success_prob, I need to fix the slacks and bounds for all those with higher probs
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] > success_prob)
                                                                      & (i + j <= n - 1))):
            slacks[i][j] = 1
            bounds[i][j] = mu

    print("\n Final objval in fixed point is ", bestobjval)
    return returnsoln, bestobjval, changed_soln, both_solns


def FixedPointPriorsIncentives(n, beta, lambd, mu, objective, transitions, incentivizedState, incentivePay, commRate):
    # after each found fixed point, find the dual prices for it, then check the revenue of the platform

    # print out the fixed point whenever you find it
    slacks, bounds, DualPrices = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    # (0, 0) should be positive, hence the slacks for all states strictly better than (0, 0) should be zero
    bounds[0][0] = mu
    minrew = 100
    for (i, j) in ((i, j) for j in range(n) for i in range(n) if (i + j <= n - 1)):
        if minrew >= objective[i][j]:
            minrew = objective[i][j]
        if objective[i][j] > objective[0][0]:
            slacks[i][j] = 1
            bounds[i][j] = mu
    print("\ncall model")
    print(bounds)

    objectivevalue = 0
    objectivevalue_b = 0
    finalobjval = 0
    changed_soln = 0
    both_solns = -1

    objValPlatform = 0
    objValPlatform_b = 0
    # argument 1 implies objective is maximize(a), argument 0 implies objective is minimize(a)
    a_soln, feasibility = OptFPPriors(n, beta, lambd, mu, bounds, slacks, objective, 1, transitions)
    if feasibility == OptimizationStatus.OPTIMAL:
        fp = a_soln
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectivevalue += objective[i][j] * fp[i][j]
        print(fp)
        FPDual = DualUseFixedPoint(n, beta, lambd, mu, fp, objective)
        print(FPDual)
        interventionPay = 0
        for i in range(len(incentivePay)):
            interventionPay = interventionPay + \
                              incentivePay[i] * fp[incentivizedState[i][0], incentivizedState[i][1]]
        objValPlatform = commRate * np.sum(np.multiply(FPDual, fp)) - interventionPay
        print("Platform's revenue is ", objValPlatform)
    print()
    a_soln_b, feasibility_b = OptFPPriors(n, beta, lambd, mu, bounds, slacks, objective, 0, transitions)
    if feasibility_b == OptimizationStatus.OPTIMAL:
        fp_b = a_soln_b
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectivevalue_b += objective[i][j] * fp_b[i][j]
        print(fp_b)
        FPDual_b = DualUseFixedPoint(n, beta, lambd, mu, fp_b, objective)
        print(FPDual_b)
        interventionPay = 0
        for i in range(len(incentivePay)):
            interventionPay = interventionPay + \
                              incentivePay[i] * fp_b[incentivizedState[i][0], incentivizedState[i][1]]
        objValPlatform_b = commRate * np.sum(np.multiply(FPDual_b, fp_b)) - interventionPay
        print("Platform's revenue is ", objValPlatform_b)

    if (feasibility == OptimizationStatus.OPTIMAL) & (feasibility_b == OptimizationStatus.OPTIMAL):
        keepsoln = fp if objValPlatform > objValPlatform_b else fp_b
        DualPrices = FPDual if objValPlatform > objValPlatform_b else FPDual_b
        print(keepsoln)
        both_solns = 1
        finalobjval -= finalobjval
        finalobjval = max(objValPlatform, objValPlatform_b)
        if abs(objectivevalue - objectivevalue_b) > 1e-6:
            changed_soln += 1
        bestobjval = finalobjval
        returnsoln = keepsoln
    elif feasibility == OptimizationStatus.OPTIMAL:
        both_solns = 0
        keepsoln = fp
        DualPrices = FPDual
        print(keepsoln)
        finalobjval -= finalobjval
        finalobjval = objValPlatform
        bestobjval = finalobjval
        returnsoln = keepsoln
    elif feasibility_b == OptimizationStatus.OPTIMAL:
        both_solns = 0
        keepsoln = fp_b
        DualPrices = FPDual_b
        print(keepsoln)
        finalobjval -= finalobjval
        finalobjval = objValPlatform_b
        bestobjval = finalobjval
        returnsoln = keepsoln
    else:
        bestobjval = -10
        returnsoln = -1 * np.ones((n, n))

    success_prob = objective[0][0]  # the success probability of (0, 0)
    counter = 0
    while (success_prob >= minrew) and (counter < n * n * 0.6):
        counter += 1
        nextsuccess_prob = 0
        ind = -1
        possible_indices = np.zeros(((n + 3) // 2, 2))
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] <= success_prob)
                                                                      & (i + j <= n - 1))):
            if objective[i][j] >= success_prob:  # to see if there are multiple subsets to be tried, if so, look at
                # their combinations, if no, just add one by one
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
                        for inc in range(ind + 1):
                            ii = int(possible_indices[inc][0])
                            jj = int(possible_indices[inc][1])
                            if s[inc] == 1:
                                bounds[ii][jj] = mu
                            else:
                                bounds[ii][jj] = 0
                        print(bounds)
                        firsttime = 0
                        for jk in range(2):
                            # print("call with multiple choices")
                            a_soln, feasibility = OptFPPriors(n, beta, lambd, mu, bounds,
                                                              slacks, objective, jk, transitions)
                            if (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 0):
                                print(a_soln)
                                FPDual = DualUseFixedPoint(n, beta, lambd, mu, a_soln, objective)
                                print(FPDual)
                                interventionPay = 0
                                for i in range(len(incentivePay)):
                                    interventionPay = interventionPay + \
                                                      incentivePay[i] * \
                                                      a_soln[incentivizedState[i][0], incentivizedState[i][1]]
                                objValPlatform = commRate * np.sum(np.multiply(FPDual, a_soln)) - interventionPay
                                print("Platform's revenue is ", objValPlatform)

                                finalobjval -= finalobjval
                                finalobjval = objValPlatform
                                if finalobjval > bestobjval:
                                    returnsoln = a_soln
                                    bestobjval = finalobjval
                                    DualPrices = FPDual
                                firsttime = 1
                            elif (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 1):
                                print(a_soln)
                                FPDual = DualUseFixedPoint(n, beta, lambd, mu, a_soln, objective)
                                print(FPDual)
                                interventionPay = 0
                                for i in range(len(incentivePay)):
                                    interventionPay = interventionPay + \
                                                      incentivePay[i] * \
                                                      a_soln[incentivizedState[i][0], incentivizedState[i][1]]
                                objValPlatform = commRate * np.sum(np.multiply(FPDual, a_soln)) - interventionPay
                                print("Platform's revenue is ", objValPlatform)

                                finalobjval -= finalobjval
                                finalobjval = objValPlatform
                                if finalobjval > bestobjval:
                                    returnsoln = a_soln
                                    bestobjval = finalobjval
                                    DualPrices = FPDual
                                    changed_soln += 1

            else:
                # print(bounds)
                ii = int(possible_indices[ind][0])
                jj = int(possible_indices[ind][1])
                bounds[ii][jj] = mu
                print(bounds)
                objectivevalue = 0
                firsttime = 0
                for jk in range(2):
                    # print("call with single choice")
                    a_soln, feasibility = OptFPPriors(n, beta, lambd, mu, bounds,
                                                      slacks, objective, jk, transitions)
                    if (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 0):
                        print(a_soln)

                        FPDual = DualUseFixedPoint(n, beta, lambd, mu, a_soln, objective)
                        print(FPDual)
                        interventionPay = 0
                        for i in range(len(incentivePay)):
                            interventionPay = interventionPay + \
                                              incentivePay[i] * \
                                              a_soln[incentivizedState[i][0], incentivizedState[i][1]]
                        objValPlatform = commRate * np.sum(np.multiply(FPDual, a_soln)) - interventionPay
                        print("Platform's revenue is ", objValPlatform)

                        finalobjval -= finalobjval
                        finalobjval = objValPlatform
                        if finalobjval > bestobjval:
                            returnsoln = a_soln
                            bestobjval = finalobjval
                            DualPrices = FPDual
                        firsttime = 1
                    elif (feasibility == OptimizationStatus.OPTIMAL) & (firsttime == 1):
                        print(a_soln)

                        FPDual = DualUseFixedPoint(n, beta, lambd, mu, a_soln, objective)
                        print(FPDual)
                        interventionPay = 0
                        for i in range(len(incentivePay)):
                            interventionPay = interventionPay + \
                                              incentivePay[i] * \
                                              a_soln[incentivizedState[i][0], incentivizedState[i][1]]
                        objValPlatform = commRate * np.sum(np.multiply(FPDual, a_soln)) - interventionPay
                        print("Platform's revenue is ", objValPlatform)

                        finalobjval -= finalobjval
                        finalobjval = objValPlatform
                        if finalobjval > bestobjval:
                            returnsoln = a_soln
                            bestobjval = finalobjval
                            DualPrices = FPDual
                            changed_soln += 1

        success_prob = nextsuccess_prob
        # before moving on to the next success_prob, I need to fix the slacks and bounds for all those with higher probs
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] > success_prob)
                                                                      & (i + j <= n - 1))):
            slacks[i][j] = 1
            bounds[i][j] = mu

    print("\n Final best Platform's revenue in fixed point is ", bestobjval)
    return returnsoln, bestobjval, changed_soln, both_solns, DualPrices


def platformModule(state, objective, commRate, workerarriveprob, wsp, jobarriveprob):
    # solves the platform's problem, i.e., maximize the commission-adjusted revenue minus incentive costs
    transitions = np.zeros((state, state))
    for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
        transitions[i][j] = (i + 1) / (i + j + 2)
    print("rewards \n", objective)
    print("and transitions between states \n", transitions)
    # without any subsidy payments
    objval_opt, jobcon, optSoln = OptPriors(state, wsp, workerarriveprob, jobarriveprob, objective, transitions)
    # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
    if jobcon - jobarriveprob > -1e-8:
        FPTree_noInt, FP_objval, solnchange, both = FixedPointPriors(state, wsp, workerarriveprob,
                                                               jobarriveprob, objective,
                                                               transitions)
        print("The fixed point is \n", FPTree_noInt)
        print("and the optimal value is ", objval_opt, "\n and LME search changed solution ", solnchange,
              " times and in the first try ", both + 1, " LME objs give a solution")

        # then get the prices for the fixed point
        FPDual_preInt = DualUseFixedPoint(state, wsp, workerarriveprob, jobarriveprob, FPTree_noInt, objective)
    platformRevenue_noSub = commRate * np.sum(np.multiply(FPDual_preInt, FPTree_noInt))

    ## objective after subsidy payments # this should be in a loop
    # for sp in range(int(state * (state - 1))):
    subObjective = deepcopy(objective)

    subObjective[0][0] = subObjective[0][0] + 0.00000002
    subObjective[0][1] = subObjective[0][1] + 0.10000001
    pos = np.nonzero(subObjective - objective)
    # print('pos', len(pos[0]), ' itself', pos)
    # raise SystemExit(0)
    pay = np.zeros(len(pos[0]))
    interState = np.zeros((len(pos[0]), 2))
    intervenedState = np.zeros((len(pos[0]), 2))
    print("\n\n\n")
    for i in range(len(pay)):
        interState[i][0], interState[i][1] = pos[0][i], pos[1][i]
        intervenedState = interState.astype(int)
        pay[i] = subObjective[intervenedState[i][0]][intervenedState[i][1]] - \
                 objective[intervenedState[i][0]][intervenedState[i][1]]
        print("Incentivizing state ", intervenedState[i][0], ",", intervenedState[i][1], " by ", pay[i])

    if np.sum(subObjective - objective) != 0:
        print("Begin subsidies\n")
        print(subObjective)
        # call the optimization model first
        objval_opt, jobcon, optSoln = OptPriors(state, wsp, workerarriveprob, jobarriveprob, subObjective, transitions)
        # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
        if jobcon - jobarriveprob > -1e-8:
            FPTree, FP_objval, solnchange, both, FPDual = FixedPointPriorsIncentives(state, wsp, workerarriveprob,
                                                                                     jobarriveprob, subObjective,
                                                                                     transitions, intervenedState,
                                                                                     pay, commRate)
            print("\nLME search changed solution ", solnchange,
                  " times and in the first try ", both + 1, " LME objs give a solution")

            platformRevenue = commRate * np.sum(np.multiply(FPDual, FPTree))
            print("Current LME is \n", FPTree)
            print("No intervention LME is \n", FPTree_noInt)
            print("And OPT (w/o interventions) is \n", optSoln)
            print("New prices \n", FPDual)
            print("Prices (w/o interventions) \n", FPDual_preInt)
            interventionPay = 0
            for i in range(len(pay)):
                interventionPay = interventionPay + \
                                  pay[i] * FPTree[intervenedState[i][0]][intervenedState[i][1]]

            print("\nplatform's return is ", platformRevenue - interventionPay,
                  " with a commission rate of", commRate * 100, "%")
            for i in range(len(pay)):
                print("subsidizing state", intervenedState[i][0], ",", intervenedState[i][1],
                      "with a payment of ", pay[i])
            print("\nand actual payment ", interventionPay, " because")
            for i in range(len(pay)):
                print("* state", intervenedState[i][0], ",", intervenedState[i][1],
                      "has mass ", FPTree[intervenedState[i][0]][intervenedState[i][1]])
        print("\nEnd subsidies")

    ### what happens without any subsidy payments ###
    print("\nIf you weren't making any subsidy payments, you would get ", platformRevenue_noSub,
          "\n  with a commission rate of ", commRate * 100, "%\n")


def policyMakerModule(state, objective):
    # solves the policy maker's problem, i.e., minimize the incentive costs to get OPT = LME

    workerarriveprob = 0.1
    wsp = 0.95
    jobarriveprob = 0.5
    transitions = np.zeros((state, state))
    for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
        transitions[i][j] = (i + 1) / (i + j + 2)
    print(objective)
    print(transitions)

    # without any subsidy payments
    objval_opt, jobcon, optSoln = OptPriors(state, wsp, workerarriveprob, jobarriveprob, objective, transitions)
    # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
    if jobcon - jobarriveprob > -1e-8:
        FPTree, FP_objval, solnchange, both = FixedPointPriors(state, wsp, workerarriveprob,
                                                               jobarriveprob, objective,
                                                               transitions)
        print("and the optimal value is ", objval_opt, "\n and LME search changed solution ", solnchange,
              " times and in the first try ", both + 1, " LME objs give a solution")

    reward_noIntervention = FP_objval
    print("No intervention reward is ", reward_noIntervention)

    subObjective = deepcopy(objective)

    subObjective[0][0] = subObjective[0][0] + 0.12
    subObjective[1][1] = subObjective[1][1] + 0.01
    pos = np.nonzero(subObjective - objective)
    pay = np.zeros(len(pos))
    intervenedState = np.zeros((len(pos), 2))
    print("\n\n\n")
    for i in range(len(pos)):
        intervenedState[i][0], intervenedState[i][1] = int(pos[0][i]), int(pos[1][i])
        pay[i] = subObjective[int(intervenedState[i][0])][int(intervenedState[i][1])] - \
                 objective[int(intervenedState[i][0])][int(intervenedState[i][1])]
        print("Incentivizing state ", int(intervenedState[i][0]), ",", int(intervenedState[i][1]), " by ", pay[i])

    if np.sum(subObjective - objective) != 0:
        print("Begin subsidies\n")
        print(subObjective)
        # call the optimization model first
        objval_opt, jobcon, optSoln = OptPriors(state, wsp, workerarriveprob, jobarriveprob, subObjective, transitions)
        # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
        if jobcon - jobarriveprob > -1e-8:
            FPTree, FP_objval, solnchange, both = FixedPointPriors(state, wsp, workerarriveprob,
                                                                   jobarriveprob, subObjective,
                                                                   transitions)
            print("\nLME search changed solution ", solnchange,
                  " times and in the first try ", both + 1, " LME objs give a solution")

            interventionPay = 0
            for i in range(len(pos)):
                interventionPay = interventionPay + \
                                  pay[i] * FPTree[int(intervenedState[i][0])][int(intervenedState[i][1])]
            if reward_noIntervention - FP_objval < 0:
                print("\nThe reward after intervention increases by ", FP_objval - reward_noIntervention,
                      "\nwith an intervention cost of ", interventionPay,
                      "\nputting the overall change at ", FP_objval - reward_noIntervention - interventionPay)
            else:
                print("\nThe reward after intervention decreases by ", reward_noIntervention - FP_objval,
                      "\nwith an intervention cost of ", interventionPay,
                      "\nputting the overall change at ", reward_noIntervention - FP_objval - interventionPay)
        print("\nEnd subsidies\n")


n = 4
lambd = 0.05
workerstayprob = 0.95
mu = 1

objVals = np.zeros((n, n))
objVals[0][0] = 0.5
objVals[0][1] = 0.4
objVals[0][2] = 0.3
objVals[0][3] = 0.2
objVals[1][0] = 0.7
objVals[1][1] = 0.6
objVals[1][2] = 0.5
objVals[2][0] = 0.8
objVals[2][1] = 0.7
objVals[3][0] = 0.9

platformModule(n, objVals, 0.065, lambd, workerstayprob, mu)

# policyMakerModule(n, objVals)
