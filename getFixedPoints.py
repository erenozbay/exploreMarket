from itertools import product
from mip import *
import numpy as np
from primals import *
import pandas as pd
import random
import time
import statistics


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
    keepsoln = a_soln
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


# def succfailFixedPoint(n, beta, lambd, mu, objective):
#     # print out the fixed point whenever you find it
#     slacks, bounds = np.zeros((n, n)), np.zeros((n, n))
#     # (0, 0) should be positive, hence the slacks for all states strictly better than (0, 0) should be zero
#     bounds[0][0] = mu
#     minrew = 1
#     for (i, j) in ((i, j) for j in range(n) for i in range(n) if (i + j <= n - 1)):
#         if minrew >= objective[i][j]:
#             minrew = objective[i][j]
#         if objective[i][j] > objective[0][0]:
#             slacks[i][j] = 1
#             bounds[i][j] = mu
#     print("call model")
#     # print(bounds)
#     a_soln, feasibility = succfailOptFixedPoint(n, beta, lambd, mu, bounds, slacks, objective, 1)
#     if feasibility == OptimizationStatus.OPTIMAL:
#         fp = a_soln
#         objectivevalue = 0
#         for (i, j) in ((i, j) for j in range(n) for i in range(n)):
#             objectivevalue += objective[i][j] * fp[i][j]
#         print(a_soln)
#
#     success_prob = objective[0][0]  # the success probability of (0, 0)
#     counter = 0
#     while (success_prob >= minrew) and (counter < n * n * 0.6):
#         counter += 1
#         nextsuccess_prob = 0
#         ind = -1
#         possible_indices = np.zeros(((n + 1) // 2, 2))
#         for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] <= success_prob)
#                                                                       & (i + j <= n - 1))):
#             if objective[i][j] >= success_prob:
#                 ind += 1
#                 possible_indices[ind][0] = i
#                 possible_indices[ind][1] = j
#             else:
#                 if objective[i][j] >= nextsuccess_prob:  # to get the next success_prob I can use
#                     nextsuccess_prob = objective[i][j]
#         if (possible_indices > 0).any():  # if you get no options above, directly move on to the next success_prob
#             # enumerate all possible solutions with the slack choices
#             # print(possible_indices)
#             if ind > 0:
#                 for s in product([0, 1], repeat=(ind + 1)):
#                     if sum(s) > 0:
#                         # print(s)
#                         # print(bounds)
#                         # print()
#                         for inc in range(ind + 1):
#                             ii = int(possible_indices[inc][0])
#                             jj = int(possible_indices[inc][1])
#                             if s[inc] == 1:
#                                 bounds[ii][jj] = mu
#                             else:
#                                 bounds[ii][jj] = 0
#                         # print(bounds)
#                         for jk in range(2):
#                             print("call with multiple choices")
#                             a_soln, feasibility = succfailOptFixedPoint(n, beta, lambd, mu, bounds,
#                                                                         slacks, objective, jk)
#                             if feasibility == OptimizationStatus.OPTIMAL:
#                                 fp = a_soln
#                                 objectivevalue = 0
#                                 for (i, j) in ((i, j) for j in range(n) for i in range(n)):
#                                     objectivevalue += objective[i][j] * fp[i][j]
#                                 print(a_soln)
#             else:
#                 # print(bounds)
#                 ii = int(possible_indices[ind][0])
#                 jj = int(possible_indices[ind][1])
#                 bounds[ii][jj] = mu
#                 # print(bounds)
#                 for jk in range(2):
#                     print("call with single choice")
#                     a_soln, feasibility = succfailOptFixedPoint(n, beta, lambd, mu, bounds, slacks,
#                                                                 objective, jk)
#                     if feasibility == OptimizationStatus.OPTIMAL:
#                         fp = a_soln
#                         objectivevalue = 0
#                         for (i, j) in ((i, j) for j in range(n) for i in range(n)):
#                             objectivevalue += objective[i][j] * fp[i][j]
#                         print(a_soln)
#
#         success_prob = nextsuccess_prob
#         # before moving on to the next success_prob, I need to fix the slacks and bounds for all those with higher probs
#         for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] > success_prob)
#                                                                       & (i + j <= n - 1))):
#             slacks[i][j] = 1
#             bounds[i][j] = mu
#
#     return fp, objectivevalue
