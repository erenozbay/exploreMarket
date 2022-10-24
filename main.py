# this is for the tree model
# from itertools import product
# from mip import *
from plotnine import *
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import time
# import warnings
import statistics
from duals import *
from primals import *
from getFixedPoints import *
from simEnvironments import *
from interventions import *
from simEnvironmentsLinear import *

pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    main()  # from simEnvironmentsLinear
    raise SystemExit(0)

    numState = 3
    workerArriveProbability = 0.1
    workerStayProbability = 0.95
    jobArriveProbability = 0.5

    objVals = np.zeros((numState, numState))
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

    track_mass, total_reward, track_queues = succfailSim(numState, TT, workerArriveProbability, jobArriveProbability,
                                                         workerStayProbability, 1000, objVals,
                                                         2 * objVals[numState - 1][0], 80)

    df_massTree = pd.DataFrame(track_mass,
                               columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                          ((i, j) for (i, j) in
                                                           product(range(numState), range(numState)) if
                                                           i + j <= (numState - 1))]], dtype=float)

    df_massTree.to_csv("massesOverTime.csv", index=False)

    # plot_mass = ggplot(df_massTree) + aes(x='Time', y=['State(0,0)', 'State(0,1)', 'State(0,2)', 'State(1,0)',
    # 'State(1,1)', 'State(2,0)']) + geom_point(size=0.005)

    # plot_mass = ggplot(aes(x="Time"), data=df_massTree) + \
    #             geom_line(aes(y='State(0,0)'), color='red') + \
    #             geom_line(aes(y='State(0,1)'), color='yellow') + \
    #             geom_line(aes(y='State(0,2)'), color='green') + \
    #             geom_line(aes(y='State(1,0)'), color='cyan') + \
    #             geom_line(aes(y='State(1,1)'), color='blue') + \
    #             geom_line(aes(y='State(2,0)'), color='purple')
    #
    # ggsave(plot_mass, filename="massConvergence.png")
    # raise SystemExit(0)

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

    # start = time.time()
    # # n = 10
    # # numsim = 1
    #
    # # simModulePriorsChange(n, numsim)
    # # simModuleDemandVarying(numsim)
    # # simModule(n, numsim)
    # # simModulePriceDev(n, numsim)
    #
    #
    # # end = time.time()
    # # print("It took ", end - start, " seconds for the whole thing")
    # # raise SystemExit(0)
    #
    # state = 3  # there are this many possible advances
    # workerarriveprob = 0.2  # at each time period, a worker arrives with this probability
    # jobarriveprob = 0.8  # at each time period, a job arrives with this probability
    # wsp = 0.99  # workerstayprobability: probability of worker staying in the system after completing a job
    #
    # rewardMultipliers = [[(i + 1) / (i + j + 2) if (i + j) < state else 0 for j in range(state)] for i in range(state)]
    #
    # rewardMultipliers[1][0] = 0.999
    # rewardMultipliers[2][0] = 0.9991
    # rewardMultipliers[1][1] = 0.00011
    # rewardMultipliers[0][0] = 0.0001
    # rewardMultipliers[0][1] = rewardMultipliers[0][0] / 2
    # rewardMultipliers[0][2] = rewardMultipliers[0][0] / 3
    #
    # print(rewardMultipliers)
    #
    # succfailDual(state, wsp, workerarriveprob, jobarriveprob, rewardMultipliers)
    #
    # # get the fixed point
    # fixedPointTree, objval_irrev = succfailFixedPoint(state, wsp, workerarriveprob, jobarriveprob, rewardMultipliers)
    # #
    # print("\nfixed point is")
    # print(fixedPointTree)
    # print()
    #
    # # # primal optimization problem
    # print("CALLING THE OPTIMIZATION MODEL")
    # xmatrix = np.zeros((state, state))
    # xmatrix_opt, obj_val_opt, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob, xmatrix, False,
    #                                                rewardMultipliers)
    # # raise SystemExit(0)
    # print()
    # xmatrix_check, obj_v, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob, xmatrix_opt, True,
    #                                            rewardMultipliers)
    # if np.abs(obj_v - obj_val_opt) > 1e-8:
    #     # warnings.warn("NOT AN LME!")
    #     print("NOT AN LME!\n")
    #
    # xmatrix[0][0] = 0.047619047619047734  # 0.066740823136818742  #
    # xmatrix[1][0] = 0.022619047619047643  # 0.031701890989988893  #
    # xmatrix[1][1] = 0.143253968235847184  # 0  #
    # xmatrix[2][0] = 0.286507936526057438  # 0.401557285873192393  #
    # # xmatrix = xmatrix_opt
    # useprevsoln = True
    # max_abs_diff = 1
    # precision = 1e-12
    # iter = 0
    # maxiter = 2000
    # while (max_abs_diff > precision) & (iter < maxiter):
    #     prev_xmatrix = deepcopy(xmatrix)
    #     xmatrix, obj_val, jobcon = succfailOpt(state, wsp, workerarriveprob, jobarriveprob, xmatrix, useprevsoln,
    #                                            rewardMultipliers)
    #     max_abs_diff = np.max(np.abs(prev_xmatrix - xmatrix))
    #     print(f"Maximum absolute difference is {max_abs_diff:.10f}")
    #     print('iter ', iter)
    #     iter += 1
    # print("Objective value is", obj_val_opt, ", Optimal soln is")
    # print(xmatrix_opt)
    # print("\nthe LME objective ratio to the optimal value is %", obj_val / obj_val_opt * 100)
    #
    # print(rewardMultipliers)
    # # raise SystemExit(0)
    #
    # # getting the dual prices using the solutions
    # print(xmatrix)
    # print("using a fixed point for the rhs")
    # succfailFixedPointDual = succfailDualUseFixedPoint(state, wsp, workerarriveprob, jobarriveprob,
    #                                                    xmatrix, rewardMultipliers)
    # print()
    # print("using the optimal solution for the rhs")
    # print(xmatrix_opt)
    # succfailFixedPointDual_wOPT = succfailDualUseFixedPoint(state, wsp, workerarriveprob, jobarriveprob,
    #                                                         xmatrix_opt, rewardMultipliers)
    #
    # # raise SystemExit(0)
    #
    # # # simulation
    # timeHorz = 500000  # number of time periods
    # bigK = 1e3
    # cC = 2 * rewardMultipliers[state - 1][0]
    # percent = 80
    # massTree, empRewardTree, queuesTree, queuesTreeCumul = succfailSim(state, timeHorz, workerarriveprob, jobarriveprob,
    #                                                                    wsp, bigK, rewardMultipliers, cC, percent)
    # # dataframes
    # df_massTree = pd.DataFrame(massTree, columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
    #                                                                 ((i, j) for (i, j) in
    #                                                                  product(range(state), range(state)) if
    #                                                                  i + j <= (state - 1))]], dtype=float)
    # df_qsTree = pd.DataFrame(queuesTree, columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
    #                                                                 ((i, j) for (i, j) in
    #                                                                  product(range(state), range(state)) if
    #                                                                  i + j <= (state - 1))]], dtype=int)
    # df_cumulQsTree = pd.DataFrame(queuesTreeCumul, columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')'
    #                                                                           for (i, j)
    #                                                                           in ((i, j) for (i, j) in
    #                                                                               product(range(state), range(state)) if
    #                                                                               i + j <= (state - 1))]], dtype=int)
    #
    # pd.set_option('display.max_columns', None)
    # print(df_massTree)
    #
    # for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
    #     name = 'State(' + str(i) + ',' + str(j) + ')'
    #     namep = 'PriceEff(' + str(i) + ',' + str(j) + ')'
    #     namep_just = 'Price(' + str(i) + ',' + str(j) + ')'
    #
    #     # local price #
    #     df_qsTree[namep_just] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    #     df_cumulQsTree[namep_just] = df_cumulQsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    #     # local price #
    #
    #     # optimal price #
    #     # name_s = 'State(' + str(i + 1) + ',' + str(j) + ')'
    #     # name_f = 'State(' + str(i) + ',' + str(j + 1) + ')'
    #     # if i + j < (state - 1):
    #     #     if (i + j == 0) or (i * j >= 1):
    #     #         df_qsTree[namep] = df_qsTree.apply(
    #     #             lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
    #     #                             wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, x[name_s])) / bigK -
    #     #                             wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, x[name_f])) / bigK), axis=1)
    #     #     elif i == 0:
    #     #         df_qsTree[namep] = df_qsTree.apply(
    #     #             lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
    #     #                             wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, x[name_f])) / bigK), axis=1)
    #     #     elif j == 0:
    #     #         df_qsTree[namep] = df_qsTree.apply(
    #     #             lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
    #     #                             wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, x[name_s])) / bigK), axis=1)
    #     # else:
    #     #     df_qsTree[namep] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    #     # df_qsTree[namep_just] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    #     # optimal price #
    #
    # # print(df_laterQsTree)
    # df_qsTree.to_csv("qs_and_localprices.csv", index=False)
    # df_cumulQsTree.to_csv("qs_and_localprices_cumulative.csv", index=False)
    # # raise SystemExit(0)
    #
    # for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
    #     name = 'State(' + str(i) + ',' + str(j) + ')'
    #     plot = ggplot(df_massTree) + aes(x='Time', y=name) + geom_line() \
    #            + geom_hline(yintercept=fixedPointTree[i][j], color="red")
    #     # + geom_hline(yintercept=xmatrix_opt[i][j], color="red")
    #     ggsave(plot, filename=name)
    #     plot_queue = ggplot(df_qsTree) + aes(x='Time', y=name) + geom_point(size=0.005)
    #     nameq = 'Queue(' + str(i) + ',' + str(j) + ')'
    #     ggsave(plot_queue, filename=nameq)
    #     # namep = 'PriceEff(' + str(i) + ',' + str(j) + ')'
    #     # plot_price = ggplot(df_qsTree) + aes(x='Time', y=namep) + geom_point(size=0.005)
    #     #              # + geom_hline(yintercept=succfailFixedPointDual[i][j], color="red")
    #     # ggsave(plot_price, filename=namep)
    #     namep_just = 'Price(' + str(i) + ',' + str(j) + ')'
    #     plot_pricej = ggplot(df_qsTree) + aes(x='Time', y=namep_just) + geom_point(size=0.005) \
    #                   + geom_hline(yintercept=succfailFixedPointDual[i][j], color="red")
    #     ggsave(plot_pricej, filename=namep_just)
    #
    # fp_obj_val = xsum(fixedPointTree[i][j] * rewardMultipliers[i][j] for j in range(state) for i in range(state))
    # print('Empirical reward is ', empRewardTree / timeHorz, ' and the reward due to the fixed point is ', fp_obj_val)
