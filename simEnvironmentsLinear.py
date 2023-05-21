from itertools import product
from mip import *
from plotnine import *
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import time
import statistics
from pulp import *
import matplotlib.pyplot as plt

# simulation for the 1-dim feedback model
def feedbackSim(n_state, T, workerArriveProb, jobArriveProb, beta, rewards, transitions, bigK, C, LOCAL_PRICES, perc,
                extraPriceAdjustment):
    # UPDATE ON 5.21.2023: This is using the transitions matrix as, element i,j is the prob of moving from i to j
    # initializations #
    recordAfter = T * perc / 100
    eps, counter_conv, tot_reward = 1e-4, 0, 0
    queue, queue_mid, track_assgn = np.zeros(n_state), np.zeros(n_state), np.zeros(n_state)
    workerarrival = np.random.binomial(1, (np.ones(T) * workerArriveProb))  # vector of arrivals for workers
    jobarrival = np.random.binomial(1, (np.ones(T) * jobArriveProb))  # vector of arrivals for jobs
    track_mass, track_queues = np.zeros((int(T / 2), n_state + 1)), np.zeros((int(T / 2), n_state + 1))
    # initializations #
    for t in range(T):
        queue[0] += workerarrival[t]
        if (jobarrival[t] == 1) & (queue.sum() > 0):  # to assign, I need a job and the system non-empty
            maxval = 0
            for i in range(n_state):
                if LOCAL_PRICES:
                    # price = (1 / (queue[i] + eps))
                    price = (bigK - min(bigK, queue[i])) / bigK
                # if using optimal prices
                else:
                    if i == 0:
                        # price = (1 / (queue[0] + eps)) - beta * (transitions[0][0] * (1 / (queue[1] + eps)))
                        price = (bigK - min(bigK, queue[0])) / bigK - \
                                beta * transitions[0][0] * (bigK - min(bigK, queue[1])) / bigK
                    elif i == (n_state - 1):
                        # price = (1 / (queue[i] + eps)) - beta * (transitions[i][2] * (1 / (queue[i - 1] + eps)))
                        price = (bigK - min(bigK, queue[i])) / bigK - \
                                beta * transitions[i][2] * (bigK - min(bigK, queue[i - 1])) / bigK
                    else:
                        # price = (1 / (queue[i] + eps)) - beta * (transitions[i][0] * (1 / (queue[i + 1] + eps))) \
                        #         - beta * (transitions[i][2] * (1 / (queue[i - 1] + eps)))
                        price = (bigK - min(bigK, queue[i])) / bigK - \
                                beta * transitions[i][0] * (bigK - min(bigK, queue[i + 1])) / bigK - \
                                beta * transitions[i][2] * (bigK - min(bigK, queue[i - 1])) / bigK
                if (maxval < (rewards[i] - C * price + extraPriceAdjustment * (1 - i))) & (queue[i] > 0):
                    maxval = rewards[i] - C * price + extraPriceAdjustment * (1 - i)
                    assigned = i
            if maxval > 0:
                queue[assigned] -= 1  # remove the assigned worker
                if (queue < 0).any():
                    print("oops, a non-existent worker left.")
                    break
                track_assgn[assigned] += 1  # keep track of the assignment, update the counter
                reward = np.random.binomial(1, rewards[assigned]) if rewards[assigned] > 0 \
                    else -1 * np.random.binomial(1, -rewards[assigned]) # sample its reward
                if t > recordAfter:
                    tot_reward += reward
                stay = np.random.binomial(1, beta)  # see if it will stay

                if stay == 1:  # if the assigned worker is staying
                    move = random.uniform(0, 1)  # decide which way it will move
                    if assigned < (n_state - 1):
                        if move <= transitions[assigned][assigned + 1]:  # it moves forward
                        # if move <= transitions[assigned][0]:  # it moves forward
                            queue[assigned + 1] += 1
                        elif (move > transitions[assigned][assigned + 1]) & \
                                (move <= transitions[assigned][assigned + 1] + transitions[assigned][assigned]):
                        # elif (move > transitions[assigned][0]) & \
                        #         (move <= transitions[assigned][0] + transitions[assigned][1]):
                            queue[assigned] += 1
                        else:  # it moves backwards
                            queue[assigned - 1] += 1
                    else:  # if the assigned worker is in the last state
                        if move <= transitions[assigned][assigned]:  # remains
                        # if move <= transitions[assigned][0] + transitions[assigned][1]:  # remains
                            queue[assigned] += 1
                        else:  # it moves backwards
                            queue[assigned - 1] += 1
        for j in range(n_state):
            queue[j] = min(bigK, queue[j])
            queue_mid[j] += queue[j]

        # keep track of assignments
        if int((t + 1) / 2) == ((t + 1) / 2):
            track_mass[counter_conv][0] = counter_conv + 1
            track_queues[counter_conv][0] = counter_conv + 1
            for j in range(n_state):
                track_mass[counter_conv][j + 1] = track_assgn[j] / (t + 1)
                track_queues[counter_conv][j + 1] = queue[j]  # queue_mid[j] / (t + 1)
            counter_conv += 1
    tot_reward = tot_reward / (T - recordAfter)
    return track_mass, tot_reward, track_queues


def feedbackOptFixedPoint(n, lambd, mu, bounds, transitions, rewardMult, beta, slacks, whichobj):
    m = LpProblem("p", LpMaximize)
    x = LpVariable.dicts("x", (range(n)), lowBound=0)
    y = LpVariable("y", 0)

    if whichobj == 1:
        m += lpSum([x[i] for i in range(n)])
    elif whichobj == 0:
        m += -y
    else:
        m += lpSum([x[i] * rewardMult[i] for i in range(n)])

    for ii in range(len(slacks)):
        j = slacks[ii]
        if j == 0:
            m += lambd + beta * transitions[0][0] * x[0] + beta * transitions[1][0] * x[1] - x[0] == 0
        elif j == (n - 1):
            m += beta * transitions[n - 2][n - 1] * x[n - 2] + \
                 beta * transitions[n - 1][n - 1] * x[n - 1] - x[n - 1] == 0
        else:
            m += beta * transitions[j - 1][j] * x[j - 1] + \
                 beta * transitions[j][j] * x[j] + beta * transitions[j + 1][j] * x[j + 1] - x[j] == 0

    m += x[0] <= lambd + beta * transitions[0][0] * x[0] + beta * transitions[1][0] * x[1]
    m += y >= lambd + beta * transitions[0][0] * x[0] + beta * transitions[1][0] * x[1] - x[0]
    m += x[n - 1] <= beta * transitions[n - 2][n - 1] * x[n - 2] + beta * transitions[n - 1][n - 1] * x[n - 1]
    m += y >= beta * transitions[n - 2][n - 1] * x[n - 2] + beta * transitions[n - 1][n - 1] * x[n - 1] - x[n - 1]
    for j in range(n):
        m += x[j] <= bounds[j]
    for j in range(1, n - 1):
        m += x[j] <= beta * transitions[j - 1][j] * x[j - 1] + \
             beta * transitions[j][j] * x[j] + beta * transitions[j + 1][j] * x[j + 1]
        m += y >= beta * transitions[j - 1][j] * x[j - 1] + \
             beta * transitions[j][j] * x[j] + beta * transitions[j + 1][j] * x[j + 1] - x[j]

    m += lpSum([x[j] for j in range(n)]) == mu
    # m.writeLP("lp.txt")

    res = not m.solve(PULP_CBC_CMD(msg=False)) == LpSolutionOptimal

    soln_sub = np.zeros(n)
    obj_ = 0
    for i in range(n):
        soln_sub[i] = value(x[i])
        obj_ += soln_sub[i] * rewardMult[i]
    return soln_sub, obj_, res


# use fixed point alg
def feedbackFixedPoint(n, lambd, mu, transition, rewardMult, beta, verbal=True):
    # I need x_0 positive, so I can move things faster if I initially start with a setting where x_0 > 0 the first time
    rewardMult_sortedIndices = np.argsort(rewardMult)[::-1]
    # start with making everybody before, and including, 0 positive
    position_0 = np.where(rewardMult_sortedIndices == 0)[0][0]
    check_feasible = True
    include_until = position_0 + 1
    soln, soln_b, soln_c = 0, 0, 0
    returnObj = -100
    returnSoln = np.zeros(n)

    while check_feasible & (include_until <= n + 1):  # solve for the variables until and including position_0
        ubs = np.zeros(n)
        force_slacks = rewardMult_sortedIndices[0:(include_until - 1)]
        for i in range(min(include_until, n)):  # give upper bounds of one for the states I want to include
            ubs[rewardMult_sortedIndices[i]] = mu
        if verbal:
            print("Upper bounds on masses", ubs)
            print("Force", force_slacks, "to have zero slacks")
        soln, obj, res = feedbackOptFixedPoint(n, lambd, mu, ubs, transition, rewardMult, beta, force_slacks, 1)
        soln_b, obj_b, res_b = feedbackOptFixedPoint(n, lambd, mu, ubs, transition, rewardMult, beta, force_slacks, 0)
        soln_c, obj_c, res_c = feedbackOptFixedPoint(n, lambd, mu, ubs, transition, rewardMult, beta, force_slacks, -1)
        if not res:
            if verbal:
                print("Maximize sums of masses", obj, end="; ")
                print(soln)
            if obj > returnObj:
                returnObj = obj
                returnSoln = soln
        else:
            if verbal:
                print("Maximize sums of masses infeasible")
        if verbal:
            print()
        if not res_b:
            if verbal:
                print("Minimize the slacks", obj_b, end="; ")
                print(soln_b)
            if obj_b > returnObj:
                returnObj = obj_b
                returnSoln = soln
        else:
            if verbal:
                print("Minimize the slacks infeasible")
        if verbal:
            print()
        if not res_c:
            if verbal:
                print("Maximize the reward", obj_c, end="; ")
                print(soln_c)
            if obj_c > returnObj:
                returnObj = obj_c
                returnSoln = soln
        else:
            if verbal:
                print("Maximize the reward infeasible")
        if verbal:
            print()
        include_until += 1

    return returnSoln, returnObj


# direct dual of the feedback model
def feedbackDual(n, lambd, mu, transitions, rewardMult, beta):
    # UPDATE ON 5.21.2023: This is using the transitions matrix as, element i,j is the prob of moving from i to j
    m = LpProblem("p", LpMinimize)
    g = LpVariable.dicts("gamma", (range(n)), lowBound=0)
    alpha = LpVariable("alpha", 0)
    m += lpSum(g[0] * lambd + alpha * mu)

    # m = Model()
    # g = [m.add_var(name='gamma({})'.format(i + 1), lb=0) for i in range(n)]
    # alpha = m.add_var(name='cap', lb=0)
    # m.objective = minimize(g[0] * lambd + alpha * mu)

    for j in range(n):
        if j == 0:
            m += alpha >= g[0] * (beta * transitions[0][0] - 1) + g[1] * (beta * transitions[0][1]) + rewardMult[0]
        elif j == (n - 1):
            m += alpha >= g[n - 2] * (beta * transitions[n - 1][n - 2]) + \
                 g[n - 1] * (beta * transitions[n - 1][n - 1] - 1) + rewardMult[n - 1]
        else:
            m += alpha >= g[j - 1] * (beta * transitions[j][j - 1]) + g[j + 1] * (beta * transitions[j][j + 1]) + \
                 g[j] * (beta * transitions[j][j] - 1) + rewardMult[j]

    # m.optimize()
    m.solve(PULP_CBC_CMD(msg=False))

    print('Optimal dual objective value is ', value(g[0]) * lambd + value(alpha) * mu)
    for i in range(n):
        print(value(g[i]), end="")
        print(", ", end="") if i < n - 1 else print(".")
    print("alpha", value(alpha))
    soln = np.zeros(n)
    for i in range(n):
        soln[i] = value(g[i])
    return soln


# dual of the feedback model using a fixed point
def feedbackDualUseFixedPoint(n, lambd, mu, transitions, rewardMult, beta, x):
    rhs = np.zeros(n)
    rhs[0] = lambd + beta * transitions[0][0] * x[0] + beta * transitions[1][0] * x[1]
    rhs[n - 1] = beta * transitions[n - 2][n - 1] * x[n - 2] + beta * transitions[n - 1][n - 1] * x[n - 1]
    for j in range(1, n - 1):
        rhs[j] = beta * transitions[j - 1][j] * x[j - 1] + beta * transitions[j][j] * x[j] + \
                 beta * transitions[j + 1][j] * x[j + 1]

    m = LpProblem("p", LpMinimize)
    g = LpVariable.dicts("gamma", (range(n)), lowBound=0)
    alpha = LpVariable("alpha", 0)
    m += lpSum([g[i] * rhs[i] for i in range(n)] + alpha * mu)

    for j in range(n):
        m += g[j] + alpha >= rewardMult[j]

    m.solve(PULP_CBC_CMD(msg=False))
    soln = np.zeros(n)
    obj = 0
    for i in range(n):
        soln[i] = value(g[i])
        obj += value(g[i]) * rhs[i]
    obj += value(alpha) * mu
    print('Dual objective value is ', obj, 'dual solution alpha is', value(alpha))
    return soln


# lp of the feedback model
def feedbackOpt(n, lambd, mu, prevSoln, usePrevSoln, transitions, rewardMult, beta):
    m = LpProblem("p", LpMaximize)
    x = LpVariable.dicts("x", (range(n)), lowBound=0)
    m += lpSum([x[i] * rewardMult[i] for i in range(n)])

    if ~usePrevSoln:
        m += x[0] <= lambd + beta * transitions[0][0] * x[0] + beta * transitions[1][0] * x[1]
        m += x[n - 1] <= beta * transitions[n - 2][n - 1] * x[n - 2] + beta * transitions[n - 1][n - 1] * x[n - 1]
        for j in range(1, n-1):
                m += x[j] <= beta * transitions[j - 1][j] * x[j - 1] + \
                     beta * transitions[j][j] * x[j] + beta * transitions[j + 1][j] * x[j + 1]
    else:
        m += x[0] <= lambd + beta * transitions[0][0] * prevSoln[0] + beta * transitions[1][0] * prevSoln[1]
        m += x[n - 1] <= beta * transitions[n - 2][n - 1] * prevSoln[n - 2] + \
             beta * transitions[n - 1][n - 1] * prevSoln[n - 1]
        for j in range(1, n - 1):
            m += x[j] <= beta * transitions[j - 1][j] * prevSoln[j - 1] + beta * transitions[j][j] * prevSoln[j] + \
                 beta * transitions[j + 1][j] * prevSoln[j + 1]

    m += lpSum([x[j] for j in range(n)]) <= mu

    m.solve(PULP_CBC_CMD(msg=False))
    soln_sub = np.zeros(n)
    obj_ = 0
    capacity = 0

    slacks = np.zeros(n)
    slacks[0] = lambd + beta * transitions[0][0] * value(x[0]) + beta * transitions[1][0] * value(x[1]) - value(x[0])
    slacks[n - 1] = beta * transitions[n - 2][n - 1] * value(x[n - 2]) + \
                    beta * transitions[n - 1][n - 1] * value(x[n - 1]) - value(x[n - 1])
    for j in range(1, n - 1):
        slacks[j] = beta * transitions[j - 1][j] * value(x[j - 1]) + beta * transitions[j][j] * value(x[j]) + \
                    beta * transitions[j + 1][j] * value(x[j + 1]) - value(x[j])

    for i in range(n):
        soln_sub[i] = value(x[i])
        obj_ += soln_sub[i] * rewardMult[i]
    print("Objective is ", obj_)
    for i in range(n):
        print("State ", i, " has ", value(x[i]), " with slack ", slacks[i])
        capacity += value(x[i])
    print("Using capacity ", capacity)

    return soln_sub, obj_, capacity


def simModuleLinear(state, numsim, workerArrivalProb = 1, jobArrivalProb = 1, wsp = 0.99):   # to get the histograms of price deviations for ML-A and ML-B instances
    willPrint = False
    wantToPlot = True
    if numsim > 5:
        willPrint = True
        print("Will print results to a csv.")
    keepRewards = np.zeros((numsim, 7))
    vocal = False
    MLmodel = "A" # "A" or "B"
    objThing = False
    if MLmodel == "B":
        objThing = True
    if objThing:
        print("Must be doing ML-B.")
    else:
        print("Must be doing ML-A.")
    ss = 0
    while ss < numsim:
        print("\nIteration", ss + 1, end=", ")
        if numsim > 1:
            workerArrivalProb = random.uniform(0.15, 0.25)
        if numsim > 1:
            jobArrivalProb = random.uniform(0.5, 0.75)
        if numsim > 1:
            wsp = random.uniform(0.8, 0.98)
        # rewardMultipliers = [-0.5, 1]
        # cC = 2 * max(rewardMultipliers)
        # transition = np.zeros((state, 3))
        # transition[0][0] = 1
        # transition[1][1] = 1
        extraPriceAdjustment = 0 # 0.75
        ############################################################################################################
        rewardMultipliers = np.array(sorted(random.sample(range(10, 100), state), reverse=objThing))
        rewardMultipliers = rewardMultipliers / (max(rewardMultipliers) + min(rewardMultipliers))
        if vocal:
            print("Rewards\n", rewardMultipliers)
            print("ML-A model!") if rewardMultipliers[0] < rewardMultipliers[1] else print("ML-B model!")
        else:
            print("ML-A model!") if rewardMultipliers[0] < rewardMultipliers[1] else print("ML-B model!")
        cC = 2 * max(rewardMultipliers)
        transition = np.zeros((state, 3))
        for j in range(1, state - 1):
            rands = np.array(sorted(random.sample(range(1, 10), 3), reverse=True))
            rands = rands / sum(rands)
            transition[j][0], transition[j][1], transition[j][2] = rands[0], rands[1], rands[2]
        rands = np.array(sorted(random.sample(range(1, 10), 2), reverse=True))
        rands = rands / sum(rands)
        transition[0][0], transition[0][1] = rands[0], rands[1]
        rands = np.array(sorted(random.sample(range(1, 10), 2), reverse=True))
        rands = rands / sum(rands)
        transition[state - 1][1], transition[state - 1][2] = rands[0], rands[1]
        # last state either remains or goes backward
        # first state either remains or goes forward
        ############################################################################################################
        # now change transitions so that it's of size state X state and each element gives the movement from state
        # i to state j
        transitionProper = np.zeros((state, state))
        # first column of transition is forward probs
        for i in range(state - 1):
            transitionProper[i][i + 1] = transition[i][0]
        # second column of transition is remain probabilities
        for i in range(state):
            transitionProper[i][i] = transition[i][1]
        # third column of transition is backward probabilities
        for i in range(1, state):
            transitionProper[i][i - 1] = transition[i][2]

        # print(transition)
        # print(transitionProper)
        transition = transitionProper
        # exit()
        ############################################################################################################

        print("lambda", workerArrivalProb, ", mu", jobArrivalProb, " beta", wsp, "\n")
        # solve for the optimal
        print("Optimal solution")
        optSoln, opt_obj_val, jobcon = feedbackOpt(state, workerArrivalProb, jobArrivalProb, np.zeros(state), False,
                                                   transition, rewardMultipliers, wsp)
        optDualSoln = feedbackDual(state, workerArrivalProb, jobArrivalProb, transition, rewardMultipliers, wsp)
        if jobcon > jobArrivalProb - 1e-8:
            # fixed point
            print("\nLooking at the fixed point")
            fixedPointSoln, obj_valFixedPoint = feedbackFixedPoint(state, workerArrivalProb, jobArrivalProb,
                                                                   transition, rewardMultipliers, wsp, verbal=False)
            print("LME/OPT is", obj_valFixedPoint / opt_obj_val)
            # obj_valFixedPoint = 0
            # for i in range(state):
            #     obj_valFixedPoint += fixedPointSoln[i] * rewardMultipliers[i]

            print("Looking at the dual prices using fixed point\nfeeding the solution", fixedPointSoln)
            fixedPointDual = feedbackDualUseFixedPoint(state, workerArrivalProb, jobArrivalProb, transition,
                                                       rewardMultipliers, wsp, fixedPointSoln)
            print("fixedPointDual prices\n", fixedPointDual)
            maxDualPrice = max(fixedPointDual)

            # simulation
            print("\nDoing the simulation")
            timeHorz = 250000  # number of time periods of each instance
            percent = 90  # exclude the first percent many iterations for reward tracking
            bigK = 1e3
            pathMass, empirical_reward, total_queues = feedbackSim(state, timeHorz, workerArrivalProb, jobArrivalProb,
                                                                   wsp, rewardMultipliers, transition, bigK, cC, True,
                                                                   percent, extraPriceAdjustment)
            df_qs = pd.DataFrame(total_queues, columns=['Time'.split() + ['State' + str(i) for i in range(state)]],
                                 dtype=float)

            if vocal:
                print("Final masses, first column is time")
                print(pathMass[-1])
                print("\nFinal queue lengths and prices after shaving earlier parts off")
            # print(df_qs)
            df_qs = df_qs.tail(int(timeHorz / 5))
            # print(df_qs)
            for i in range(state):
                name = 'State' + str(i)
                df_qs[name] = df_qs[name].astype(int)
                namep = 'Price' + str(i)
                df_qs[namep] = df_qs.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
            if vocal:
                print(df_qs)
            print("\n The simulated prices")
            pricesFromSim = np.zeros(state)
            pricesFromSimActual = np.zeros(state)
            for i in range(state):
                name = 'State' + str(i)  # 'Price' + str(i)
                mid = df_qs[name]
                # print(mid)
                midd = mid[mid.columns[0]].values
                # print(midd)
                # pricesFromSimActual[i] = statistics.mean(midd[-int((timeHorz * (1 - percent / 100)) / 2 - 1):])
                pricesFromSimActual[i] = cC * (1 - min(1, statistics.mean(midd[-int((timeHorz * (1 - percent / 100)) / 2 - 1):]) / bigK))
                if fixedPointDual[i] != 0:
                    pricesFromSim[i] = pricesFromSimActual[i]

            print("Actual prices from simulation", pricesFromSimActual)
            print("Prices I will use to compare the LME", pricesFromSim)
            # print(fixedPointDual)
            # sth = False
            # for jj in range(state):
            #     if fixedPointDual[jj] == 0:
            #         sth = True
            #     if sth:
            #         pricesFromSim[jj] = 0
            # print(pricesFromSim)
            diffprices = np.abs(fixedPointDual - pricesFromSim)
            maxdiff = diffprices.max()
            # pos = np.argmax(diffprices, axis=None)
            keepRewards[ss][0] = opt_obj_val
            keepRewards[ss][1] = empirical_reward
            keepRewards[ss][2] = obj_valFixedPoint / opt_obj_val
            keepRewards[ss][3] = maxdiff / maxDualPrice
            keepRewards[ss][4] = workerArrivalProb
            keepRewards[ss][5] = jobArrivalProb
            keepRewards[ss][6] = wsp
            ss += 1
            print(keepRewards[:ss,])
            if wantToPlot and keepRewards[ss - 1][3] > 0.2:
                plt.figure(figsize=(7, 5), dpi=100)
                plt.rc('axes', axisbelow=True)
                plt.grid(lw=1.1)
                plt.plot(df_qs['Time'].to_numpy(), df_qs['Price0'].to_numpy(), color='green',
                         label='State 0')
                plt.plot(df_qs['Time'].to_numpy(), df_qs['Price1'].to_numpy(), color='red',
                         label='State 1')
                plt.plot(df_qs['Time'].to_numpy(), df_qs['Price2'].to_numpy(), color='blue',
                         label='State 2')
                plt.plot(df_qs['Time'].to_numpy(), df_qs['Price3'].to_numpy(), color='purple',
                         label='State 3')
                plt.ylabel('Prices')
                plt.xlabel('Time')
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
                plt.tight_layout()
                # plt.show()
                plt.savefig("newFeedbackFigs/instance" + str(ss - 1) + "_priceRatio" + str(keepRewards[ss - 1][3])[:6], format='eps', bbox_inches='tight')
                plt.cla()
        else:
            print("\nIter ", ss + 1, " won't give a fixed point b/c the capacity constraint not tight, try again\n")
    if willPrint:
        np.savetxt("feedbackLMEvsOPTobjvalsAndPriceRatiosML-" + MLmodel + "_" + str(numsim) + "sims" +
                   ".csv", keepRewards, delimiter=",")


def main():
    start = time.time()
    # state = 2  # there are this many states in total
    # numsims = 50
    # simModule(state, numsims)
    # raise SystemExit(0)

    # # still large price diff with T = 5M and bigK = 1e4, b/c not enough accumulation in state with theoretical price = 0
    # # which is state 2, price0 = 0.29090909, price1 = 0.08181818; x0 = 0.31798, x1 = 0.26721, x2 = 0.14772
    # # this doesn't really work with smaller or larger bigK and longer sim, I guess beta is too small
    # # the example below this one shows that beta can be dealt with using a smaller bigK
    # # T = 5M and bigK = 1e3 is okay; T = 2M and bigK = 1e3 does not cut it, simulated price0 is still too high
    # T = 5M and bigK = 5e3 is not good, 10% off, needs longer
    # rewards = [0.75454545, 0.54545455, 0.46363636, 0.32727273, 0.24545455]
    # transition = [[0.66666667, 0.33333333, 0],
    #               [0.5, 0.3125, 0.1875],
    #               [0.42857143, 0.33333333, 0.23809524],
    #               [0.5, 0.41666667, 0.08333333],
    #               [0.64285714, 0.21428571, 0.14285714]]
    # lambd = 0.1918405227361316
    # muu = 0.7329132945412744
    # beta = 0.8081096159529555

    ################## these two are similar in nature, but their fixes are opposite ##################
    # # T = 2M and bigK = 1e4 doesn't work, price3 = 0 but simulated price3 > 0 and everybody is shifted by that amount
    # # price0 = 0.36190476, price1 = 0.11428571, price2 = 0.08571428
    # # x0 = 0.26255719, x1 =  0.18525647, x2 = 0.08721418, x3 = 0.03656579, x4 = 0
    # # T = 1M and bigK = 1e3 works well, so maybe beta is not the main culprit
    # rewards = [0.8, 0.55238095, 0.52380952, 0.43809524, 0.2]
    # transition = [[0.55555556, 0.44444444, 0],
    #               [0.42857143, 0.35714286, 0.214285711],
    #               [0.5, 0.28571429, 0.214285711],
    #               [0.52941176, 0.35294118, 0.11764706],
    #               [0.39130435, 0.34782609, 0.2608695711]]
    # lambd = 0.13698232960816656
    # muu = 0.571593636687612
    # beta = 0.8029601749722973
    # T = 1M, bigK = 1e3 is not good because price0 = 0.01, price1 = 0 -- small movements in x1 queue causes simulated
    # price1 to go above zero and so price0 = 0.01 + simPrice1
    # T = 1M, bigK = 1e4 alleviates this a bit, price1 is a lot closer to zero, T = 2M is obviously better
    # T = 5M, bigK = 1e3 doesn't work, price0 = 0.015
    # T = 5M, bigK = 5e3 is okay-ish, price0 = 0.0113
    # rewards = [0.85, 0.84, 0.62, 0.38, 0.15]
    # transition = [[0.9, 0.1, 0],
    #               [0.5, 0.42857143, 0.07142857],
    #               [0.5, 0.4, 0.1],
    #               [0.53333333, 0.4, 0.06666667],
    #               [0.375, 0.33333333, 0.29166667]]
    # lambd = 0.24030096207304935
    # muu = 0.5547624676429614
    # beta = 0.8304121604778487
    ################## these two are similar in nature, but their fixes are opposite ##################

    state = 3
    rewards = [0.11, 0.1, 1]
    transition = [[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 1]]
    lambd = 0.75
    muu = 1
    beta = 0.5

    optSoln, opt_obj_val, _ = feedbackOpt(state, lambd, muu, np.zeros(state), False, transition, rewards, beta)
    print("Looking at the fixed point")
    fixedPointSoln = feedbackFixedPoint(state, lambd, muu, transition, rewards, beta)
    print(fixedPointSoln)
    print("Looking at the dual prices using fixed point")
    fixedPointDual = feedbackDualUseFixedPoint(state, lambd, muu, transition, rewards, beta, fixedPointSoln)
    print(fixedPointDual)
    raise SystemExit(0)

    cC = 2 * max(rewards)
    timeHorz = 5000000  # number of time periods of each instance
    percent = 80  # exclude the first percent many iterations for reward tracking
    bigK = 5e3
    pathMass, empirical_reward, total_queues = feedbackSim(state, timeHorz, lambd, muu,
                                                           beta, rewards, transition, bigK, cC, True,
                                                           percent)
    print("Final masses")
    print(pathMass[-1])
    df_qs = pd.DataFrame(total_queues, columns=['Time'.split() + ['State' + str(i) for i in range(state)]],
                         dtype=float)
    df_qs = df_qs.tail(int(timeHorz / 5))
    print(df_qs)
    for i in range(state):
        name = 'State' + str(i)
        df_qs[name] = df_qs[name].astype(int)
        namep = 'Price' + str(i)
        df_qs[namep] = df_qs.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
    print(df_qs)
    # df_qs.to_csv("gettingcorrectprices.csv", index=False)
    mid = df_qs['Price0']
    midd = mid[mid.columns[0]].values
    # np.savetxt("this.csv", midd[-int((timeHorz * (1 - percent / 100)) / 10):], delimiter=",")
    print("Price of 0 ", statistics.mean(midd[-int((timeHorz * (1 - percent / 100)) / 2 - 1):]))

    end = time.time()
    print("It took ", end - start, " seconds for the whole thing")
    raise SystemExit(0)

    ##### simulation module #####
    state = 5  # there are this many states in total
    ## update lambda, mu and beta as well and get different runs?
    workerArrivalProb = 0.1  # lambda: at each time period, a new worker arrives with this probability
    jobArrivalProb = 0.5  # mu: at each time period, a job arrives with this probability
    wsp = 0.95  # workerstayprobability: probability of worker staying in the system after completing a job

    rewardMultipliers = np.array(sorted(random.sample(range(10, 100), state), reverse=True))
    rewardMultipliers = rewardMultipliers / (max(rewardMultipliers) + min(rewardMultipliers))
    print(rewardMultipliers)
    cC = 2 * max(rewardMultipliers)
    transition = np.zeros((state, 3))
    for j in range(1, state):
        rands = np.array(sorted(random.sample(range(1, 10), 3), reverse=False))
        rands = rands / sum(rands)
        transition[j][0], transition[j][1], transition[j][2] = rands[0], rands[1], rands[2]
    rands = np.array(sorted(random.sample(range(1, 10), 2), reverse=False))
    rands = rands / sum(rands)
    transition[0][0], transition[0][1] = rands[0], rands[1]
    print(transition)

    sims = 100  # number of different runs
    timeHorz = 250000  # number of time periods of each instance
    percent = 80  # exclude the first percent many iterations for reward tracking
    bigK = 1e3
    empRewards = np.zeros(sims)
    # solve for the optimal
    optSoln, opt_obj_val = feedbackOpt(state, workerArrivalProb, jobArrivalProb, np.zeros(state), False,
                                       transition, rewardMultipliers, wsp)

    for i in range(sims):
        pathMass, empirical_reward, total_queues = feedbackSim(state, timeHorz, workerArrivalProb, jobArrivalProb,
                                                               wsp, rewardMultipliers, transition, bigK, cC, True, percent)
        empRewards[i] = empirical_reward
        # df_mass = pd.DataFrame(pathMass, columns=['Time'.split() + ['State' + str(i) for i in range(state)]], dtype=float)
        # df_mass['Time'] = df_mass['Time'].astype(int)
        # for j in range(state):
        #     name = 'State' + str(j)
        #     plot = ggplot(df_mass) + aes(x='Time', y=name) + geom_line() + geom_hline(yintercept=optSoln[j], color="red")
        #     name = "T" + str(i) + ', State' + str(j)
        #     ggsave(plot, filename=name)
        print("Iter ", i + 1)
        print(pathMass[-1])
        print(total_queues[-1])
    print()
    print("Optimal reward is ", opt_obj_val)
    print("Empirical rewards are")
    print(empRewards)
    diff1 = (opt_obj_val - empRewards) / opt_obj_val * 100
    print("Empirical rewards compared to the optimal reward (in terms of percentage difference):")
    print(diff1)
    diff2 = empRewards / opt_obj_val * 100
    print("\n Empirical rewards compared to the optimal reward (in terms of ratio to optimal value):")
    print(diff2)
    ##### simulation module #####
    end = time.time()
    print("It took ", end - start, " seconds for the whole thing")
    print("Rewards are")
    print(rewardMultipliers)
    print("Transitions are")
    print(transition)
    raise SystemExit(0)

    state = 5  # there are this many states in total
    timeHorz = 10000000  # number of time periods
    workerArrivalProb = 0.1  # lambda: at each time period, a new worker arrives with this probability
    jobArrivalProb = 0.45  # mu: at each time period, a job arrives with this probability
    wsp = 0.95  # workerstayprobability: probability of worker staying in the system after completing a job
    bigK = 1e3
    usingLocalPrices = True
    # rewardMultipliers = [(i + 1) / (i + 2) for i in range(state)]
    rewardMultipliers = [(i + 3) / (2 * i + 4) for i in range(state)]
    # rewardMultipliers = [((state / 3 - (state / 2 - i) * (state / 2 - i) / state) / (state / 3))
    #                      / (i % 2 + 1) for i in range(state)]
    # rewardMultipliers[0] = 0.8
    print(rewardMultipliers)
    cC = 2 * max(rewardMultipliers)  # / (1 - wsp)
    transition = np.zeros((state, 3))
    transition[0][0], transition[0][1] = 1 / 2, 1 / 2
    transition[state - 1][0], transition[state - 1][1], transition[state - 1][2] = 1 / 2, 3 / 8, 1 / 8
    for i in range(1, state - 1):
        transition[i][0], transition[i][1] = 1 / 2, 1 / 3
        transition[i][2] = 1 - (transition[i][0] + transition[i][1])
    print(transition)
    print()

    # first call the algorithm to find a/the fixed point
    print('doing alg results')
    fixedPoint = feedbackFixedPoint(state, workerArrivalProb, jobArrivalProb, transition, rewardMultipliers, wsp)
    # print(fixedPoint)
    # raise SystemExit(0)
    obj_val = 0
    for i in range(state):
        obj_val += fixedPoint[i] * rewardMultipliers[i]
    print()

    # then solve the dual for the fixed point
    print('doing fixed point dual results')
    fixedPointDual = feedbackDualUseFixedPoint(state, workerArrivalProb, jobArrivalProb, transition,
                                               rewardMultipliers, wsp, fixedPoint)
    print(fixedPointDual)
    print()

    # then call the optimization problem to find a/the fixed point using it
    print('solving the optimization problem')
    usePreviousSoln = False
    previousSoln = np.zeros(state)
    tolerance = 1e-8
    max_abs_diff = 1
    max_iter = 10
    iteration = 0
    # loop over the iterations #
    while (max_abs_diff > tolerance) & (iteration < max_iter):
        print("iteration ", iteration + 1)
        optSoln, opt_obj_val = feedbackOpt(state, workerArrivalProb, jobArrivalProb, previousSoln, usePreviousSoln,
                                           transition, rewardMultipliers, wsp)
        max_abs_diff = np.max(np.abs(previousSoln - optSoln))
        previousSoln = deepcopy(optSoln)
        usePreviousSoln = True
        iteration += 1
    print("Did ", iteration, " iterations")
    # loop over the iterations #

    raise SystemExit(0)
    print('do the simulation')
    pathMass, empirical_reward, total_queues = feedbackSim(state, timeHorz, workerArrivalProb, jobArrivalProb,
                                                           wsp, rewardMultipliers, transition, bigK, cC, usingLocalPrices)
    # dataframes
    df_mass = pd.DataFrame(pathMass, columns=['Time'.split() + ['State' + str(i) for i in range(state)]], dtype=float)
    df_qs = pd.DataFrame(total_queues, columns=['Time'.split() + ['State' + str(i) for i in range(state)]], dtype=float)
    print()
    print('Fixed point objective value is ', obj_val, 'Empirical average reward is ', empirical_reward / timeHorz,
          ", off by %.4f%%" % (np.abs(obj_val - empirical_reward / timeHorz) / obj_val * 100))
    print()
    df_mass['Time'] = df_mass['Time'].astype(int)
    df_qs['Time'] = df_qs['Time'].astype(int)
    print(df_qs)
    for i in range(state):
        name = 'State' + str(i)
        df_qs[name] = df_qs[name].astype(int)
        namep = 'Price' + str(i)
        if usingLocalPrices:
            df_qs[namep] = df_qs.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
        else:
            name_p = 'State' + str(i + 1)
            name_m = 'State' + str(i - 1)
            if i == 0:
                df_qs[namep] = df_qs.apply(lambda x: cC * ((bigK - min(bigK, x[name])) / bigK - wsp *
                                                           transition[i][0] * (bigK - min(bigK, x[name_p])) / bigK), axis=1)
            elif i == (state - 1):
                df_qs[namep] = df_qs.apply(lambda x: cC * ((bigK - min(bigK, x[name])) / bigK - wsp *
                                                           transition[i][2] * (bigK - min(bigK, x[name_m])) / bigK), axis=1)
            else:
                df_qs[namep] = df_qs.apply(lambda x: cC * ((bigK - min(bigK, x[name])) / bigK - wsp *
                                                           transition[i][0] * (bigK - min(bigK, x[name_p])) / bigK - wsp *
                                                           transition[i][2] * (bigK - min(bigK, x[name_m])) / bigK), axis=1)
        namepap = 'PriceAdjPayoff' + str(i)
        df_qs[namepap] = rewardMultipliers[i] - df_qs[namep]

    df_qs_sub = df_qs[(df_qs['Time'] > 1000).to_numpy()]
    # print(df_mass)
    print(df_qs_sub)

    # ggplot part
    for i in range(state):
        name = 'State' + str(i)
        plot = ggplot(df_mass) + aes(x='Time', y=name) + geom_line() + geom_hline(yintercept=fixedPoint[i], color="red")
        ggsave(plot, filename=name)
        plot_queue = ggplot(df_qs_sub) + aes(x='Time', y=name) + geom_point(size=0.005)
        nameq = 'Queue' + str(i)
        ggsave(plot_queue, filename=nameq)
        namep = 'Price' + str(i)
        plot_price = ggplot(df_qs_sub) + aes(x='Time', y=namep) + geom_point(size=0.005) + \
                     geom_hline(yintercept=fixedPointDual[i], color="red")
        ggsave(plot_price, filename=namep)
        namepap = 'PriceAdjPayoff' + str(i)
        plot_pap = ggplot(df_qs_sub) + aes(x='Time', y=namepap) + geom_point(size=0.005)
        ggsave(plot_pap, filename=namepap)

    # dd = df_qs_sub.iloc[-1:].to_numpy()
    # for i in range(len(dd)):
    #     print(dd[i])
    # trying matplotlib
    # giving ValueError: x must be a label or position, talking about Time column, what??
    # df_mass.plot(kind='line', x='Time', y='State0')
    # plt.show()
