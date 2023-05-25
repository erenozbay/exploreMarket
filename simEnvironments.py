# this is for the tree model
from itertools import product
from mip import *
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


# considers the parameters in beta as the position, (success, failure), with some room for deviating those with iii, jjj
def succfailSim(state, T, workerarriveprob, jobarriveprob, wsp, bigK, rewardprob, C, percent, recordEvery = 10,
                optimalOrLocal = 'local', iii=0, jjj=0):
    counter_conv, total_reward, counterr = 0, 0, 0
    queue, track_assign, queue_mid = np.zeros((state, state)), np.zeros((state, state)), np.zeros((state, state))
    track_mass, track_queues, track_queues_cum = np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1)), \
                                                 np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1)), \
                                                 np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1))
    # last_queues = np.zeros((int(T * (1 - percent / 100)), int((state + 1) * state * 0.5) + 1))
    # pricesHere = np.zeros((state, state))
    ####

    workerarrival = np.random.binomial(1, (np.ones(T) * workerarriveprob))  # vector of arrivals for workers
    if jobarriveprob <= 1:
        jobarrival = np.random.binomial(1, (np.ones(T) * jobarriveprob))  # vector of arrivals for jobs
    else:
        intPart = np.floor(jobarriveprob)
        jobarrival = np.ones(T) * intPart + np.random.binomial(1, (np.ones(T) * (jobarriveprob - intPart)))

    print("total arrivals will be ", jobarrival.sum())

    for t in range(T):
        queue[0][0] += workerarrival[t]
        activeJobs = jobarrival[t]
        while activeJobs > 1 and queue.sum() > 0:
            activeJobs -= 1
        # if (jobarrival[t] >= 1) & (queue.sum() > 0):  # to assign, I need a job and the system non-empty
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
                if optimalOrLocal == 'optimal':
                    if i + j < (state - 1):
                        # if (i + j == 0) or (i * j >= 1):
                        price = (bigK - min(bigK, queue[i][j])) / bigK - \
                                    wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, queue[i + 1][j])) / bigK - \
                                    wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, queue[i][j + 1])) / bigK
                        # elif i == 0:
                        #     price = (bigK - min(bigK, queue[i][j])) / bigK - \
                        #             wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, queue[i][j + 1])) / bigK
                        # elif j == 0:
                        #     price = (bigK - min(bigK, queue[i][j])) / bigK - \
                        #             wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, queue[i + 1][j])) / bigK
                    elif i + j == (state - 1):
                        price = (bigK - min(bigK, queue[i][j])) / bigK - \
                                wsp * (bigK - min(bigK, queue[i][j])) / bigK
                else:
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
                # success = rewardprob[i][j]
                reward_param = np.random.beta(pos_i + iii + 1, pos_j + jjj + 1)
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

        if int((t + 1) / recordEvery) == ((t + 1) / recordEvery):
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
    # pricesHere = pricesHere / (T * (1 - percent / 100))
    # print(track_queues[-1, :])
    return track_mass, total_reward, track_queues


def succfailLinear(state, T, workerarriveprob, jobarriveprob, wsp, bigK, rewardList, transition,
                   initQueue, C, percent, recordEvery=10, optimalOrLocal = 'local'):
    # only deals with forward transitions, no remain for any state except the last, no backflow for anybody
    counter_conv, total_reward, counterr = 0, 0, 0
    queue, track_assign, queue_mid = initQueue, np.zeros(state), np.zeros(state)

    # +1 in col is for time
    track_mass, track_queues = np.zeros((int(T / recordEvery), state + 1)), np.zeros((int(T / recordEvery), state + 1))
    # last_queues = np.zeros((int(T * (1 - percent / 100)), int((state + 1) * state * 0.5) + 1))
    # pricesHere = np.zeros((state, state))
    lengh=0
    ####

    workerarrival = np.random.binomial(1, (np.ones(T) * workerarriveprob))  # vector of arrivals for workers
    # jobarrival = np.ones(T) * jobarriveprob #+ np.random.binomial(1, (np.ones(T) * 0.1))
    jobarrival = np.random.binomial(1, (np.ones(T) * jobarriveprob))  # vector of arrivals for jobs
    print("total arrivals will be ", jobarrival.sum())

    for t in range(T):
        queue[0] += workerarrival[t]
        if (jobarrival[t] >= 1) & (queue.sum() > 0):  # to assign, I need a job and the system non-empty
            maxval = 0
            firstTime = 0
            pickFrom = []

            for i in range(state):
                if optimalOrLocal == 'local':
                    price = (bigK - min(bigK, queue[i])) / bigK
                elif optimalOrLocal == 'optimal':
                    if i < state - 1:
                        price = (bigK - min(bigK, queue[i])) / bigK - \
                                wsp * transition[i][i + 1] * (bigK - min(bigK, queue[i + 1])) / bigK
                    else:
                        price = (bigK - min(bigK, queue[i])) / bigK * (1 - wsp)
                else:
                    exit("Choose either optimal or local pricing algorithm.")
                if (maxval <= (rewardList[i] - C * price)) & (queue[i] > 0):
                    if firstTime == 0:
                        maxval = rewardList[i] - C * price
                        pickFrom.append(i)
                        firstTime = 1
                    else:
                        if maxval <= (rewardList[i] - C * price):
                            prev_maxval = maxval
                            maxval = rewardList[i] - C * price
                            if prev_maxval == maxval:
                                pickFrom.append(i)
                            else:
                                pickFrom.clear()
                                pickFrom.append(i)

            if maxval > 0:
                pos_i = random.choice(pickFrom)
                lengh += 1 if len(pickFrom) > 1 else 0
                queue[pos_i] -= 1
                if (queue < 0).any():
                    print("oops, a non-existent worker left.")
                    break
                track_assign[pos_i] += 1

                reward = np.random.binomial(1, rewardList[pos_i])
                if t > T * percent / 100:
                    total_reward += reward
                stay = np.random.binomial(1, wsp)  # see if it will stay

                if stay == 1:  # if the assigned worker is staying
                    if pos_i < (state - 1):
                        queue[pos_i + 1] += 1
                    else:
                        queue[pos_i] += 1

        for i in range(state):
            queue[i] = min(queue[i], bigK)
            queue_mid[i] += queue[i]


        if int((t + 1) / recordEvery) == ((t + 1) / recordEvery):
            track_mass[counter_conv][0] = counter_conv + 1
            track_queues[counter_conv][0] = counter_conv + 1
            # track_queues_cum[counter_conv][0] = counter_conv + 1
            index = 0
            for i in range(state):
                track_mass[counter_conv][index + 1] = track_assign[i] / (t + 1)
                track_queues[counter_conv][index + 1] = queue[i]  # queue_mid[i][j] / (t + 1)
                # track_queues_cum[counter_conv][index + 1] = queue_mid[i][j] / (t + 1)
                index += 1
            counter_conv += 1
    total_reward = total_reward / (T * (1 - percent / 100))
    # pricesHere = pricesHere / (T * (1 - percent / 100))
    # print(track_mass)
    # print(lengh)
    return track_mass, total_reward, track_queues



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
                    stateName = 'State(' + str(i) + ',' + str(j) + ')'
                    namep_just = 'Price(' + str(i) + ',' + str(j) + ')'

                    # local price #
                    df_qsTree[namep_just] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[stateName])) / bigK,
                                                            axis=1)

                for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                    stateName = 'Price(' + str(i) + ',' + str(j) + ')'
                    mid = df_qsTree[stateName]
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
    # simulation module####
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
    np.savetxt("keepRewards.csv", keepRewards[0:index], delimiter=",")


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


def succfailSim_alt(state, T, workerarriveprob, jobarriveprob, wsp, bigK, rewardprob, C, percent, recordEvery = 10,
                optimalOrLocal = 'local', iii=0, jjj=0):
    counter_conv, total_reward, counterr = 0, 0, 0
    queue, track_assign, queue_mid = np.zeros((state, state)), np.zeros((state, state)), np.zeros((state, state))
    track_mass, track_queues, track_queues_cum = np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1)), \
                                                 np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1)), \
                                                 np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1))

    workerarrival = np.random.binomial(1, (np.ones(T) * workerarriveprob))  # vector of arrivals for workers
    print("total worker arrivals will be", workerarrival.sum())
    if jobarriveprob <= 1:
        jobarrival = np.random.binomial(1, (np.ones(T) * jobarriveprob))  # vector of arrivals for jobs
    else:
        intPart = np.floor(jobarriveprob)
        jobarrival = np.ones(T) * intPart + np.random.binomial(1, (np.ones(T) * (jobarriveprob - intPart)))

    print("total job arrivals will be", jobarrival.sum())

    def getPrices(queueue):
        price = deepcopy(queueue)
        for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
            if optimalOrLocal == 'optimal':
                if i + j < (state - 1):
                    price[i][j] = (bigK - min(bigK, queueue[i][j])) / bigK - \
                            wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, queueue[i + 1][j])) / bigK - \
                            wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, queueue[i][j + 1])) / bigK
                elif i + j == (state - 1):
                    price[i][j] = (bigK - min(bigK, queueue[i][j])) / bigK - \
                            wsp * (bigK - min(bigK, queueue[i][j])) / bigK
            else:
                price[i][j] = (bigK - min(bigK, queueue[i][j])) / bigK
        return price

    def getPos(pricesForAlll, queueueue, maxvall=0, randomize=0):
        pos_ii, pos_jj = -1, -1
        for (ii, jj) in ((ii, jj) for (ii, jj) in product(range(state), range(state)) if ii + jj <= (state - 1)):
            priceHere = pricesForAlll[ii][jj]
            if (maxvall <= (rewardprob[ii][jj] - C * priceHere)) & (queueueue[ii][jj] > 0):
                if maxvall < (rewardprob[ii][jj] - C * priceHere):  # to randomize selections of, e.g., (1,1) and (2,2)
                    randomize = 1
                    maxvall = rewardprob[ii][jj] - C * priceHere
                    pos_ii = ii
                    pos_jj = jj
                elif randomize > 0:
                    if rewardprob[ii][jj] == rewardprob[pos_ii][pos_jj]:
                        choose = random.uniform(0, 1)
                        if choose >= 0.5:
                            pos_ii = ii
                            pos_jj = jj
        return {'i': pos_ii, 'j': pos_jj, 'maxval': maxvall}

    activeJobs = 0
    for t in range(T):
        queue[0][0] += workerarrival[t]
        activeJobs += jobarrival[t]
        if (jobarrival[t] >= 1) & (queue.sum() > 0):  # to assign, I need a job and the system non-empty
            # maxval = 0

            # pricesForAll = getPrices(queue)
            tryNow = True
            while activeJobs >= 1 and queue.sum() > 0 and tryNow:
                pricesForAll = getPrices(queue)
                twoPos = getPos(pricesForAll, queue)
                pos_i = twoPos['i']
                pos_j = twoPos['j']
                maxval = twoPos['maxval']
                # print("activeJobs", activeJobs, "t", t)

                # for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                #     price = pricesForAll[i][j]
                #     if (maxval <= (rewardprob[i][j] - C * price)) & (queue[i][j] > 0):
                #         if maxval < (rewardprob[i][j] - C * price):  # to randomize selections of, e.g., (1,1) and (2,2)
                #             randomize = 1
                #             maxval = rewardprob[i][j] - C * price
                #             pos_i = i
                #             pos_j = j
                #         elif randomize > 0:
                #             if rewardprob[i][j] == rewardprob[pos_i][pos_j]:
                #                 choose = random.uniform(0, 1)
                #                 if choose >= 0.5:
                #                     pos_i = i
                #                     pos_j = j

                if maxval > 0:
                    activeJobs -= 1
                    if (queue < 0).any():
                        print("oops, a non-existent worker left.")
                        break
                    # else:
                    #     print("All good")
                    queue[pos_i][pos_j] -= 1
                    track_assign[pos_i][pos_j] += 1
                    # success = rewardprob[i][j]
                    reward_param = np.random.beta(pos_i + iii + 1, pos_j + jjj + 1)
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
                else:
                    tryNow = False
                # maxval = 0

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

        if int((t + 1) / recordEvery) == ((t + 1) / recordEvery):
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
    # pricesHere = pricesHere / (T * (1 - percent / 100))
    # print(track_queues[-1, :])
    print("Final number of available jobs", activeJobs)
    return track_mass, total_reward, track_queues


def simModulePriorsChange(state, sims):
    T = 250000 # time
    wap = 0.5
    wsp = 0.95
    alphas = [(9 - i) / (1 * 10) for i in range(9)]
    keepResults = np.zeros((len(alphas), 4))
    simstart, simend = 0, 0
    start = time()
    for ss in range(len(alphas)):
        jobarriveprob = 10 * alphas[ss] + 0.1
        print("jobarriveprob", jobarriveprob)
        # jobarriveprob = wap / (1 - wsp) * alphas[ss]  # alphas[len(alphas) - ss - 1]
        # jobarriveprob = jobarriveprob if jobarriveprob <= 1 else 1

        # keepMid = np.zeros((int(state * (state + 1) / 2), 3))
        priorstuff = 5
        keepMid = np.zeros((priorstuff, 3))
        ind = 0
        # for (ii, jj) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
        #     zeroprior_s = ii
        #     zeroprior_f = jj
        for jj in range(priorstuff):
            zeroprior_f = (priorstuff - 1) - jj
            print("Prior is (1," + str(zeroprior_f + 1) +")")
            # get the objective, each element keeps the ACTUAL success probability
            objective = np.zeros((state, state))
            for (iii, jjj) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                objective[iii][jjj] = (iii + 1) / (iii + jjj + zeroprior_f + 2)
            # print(objective)
            transition = {}
            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                transition[str(i) + "_" + str(j)] = {}
                if i + j < (state - 1):
                    transition[str(i) + "_" + str(j)]['s'] = objective[i][j]  # success
                    transition[str(i) + "_" + str(j)]['f'] = 1 - transition[str(i) + "_" + str(j)]['s']  # failure
                    transition[str(i) + "_" + str(j)]['r'] = 0  # remain
                else:
                    transition[str(i) + "_" + str(j)]['s'] = 0  # success
                    transition[str(i) + "_" + str(j)]['f'] = 0  # failure
                    transition[str(i) + "_" + str(j)]['r'] = 1
                # print(i, j, "", transition[str(i) + str(j)].values())
                if sum(transition[str(i) + "_" + str(j)].values()) != 1:
                    exit("problem in (" + str(i) + ", " + str(j) + ")")
            # call the optimization model first
            _, _, objval_opt, jobcon = succfailOpt(state, wsp, wap, jobarriveprob, np.zeros((state, state)), False,
                                                   objective, transition, np.ones((state, state)) * 10, printResults=False)
            # succfailOptPriors(state, wsp, wap, jobarriveprob, objective)  # this was used right above
            # if fixed point is doable, i.e., job constraint is fully utilized, go on to the fixed point
            FP_objval = 0
            simsRewAvg = 0
            print("The optimal value is", objval_opt, "and the unused portion of mu", jobarriveprob - jobcon)
            if jobcon - jobarriveprob > -1e-7:
                FPTree, FP_objval, solnchange, both = succfailFixedPoint(state, wsp, wap, jobarriveprob,
                                                                         objective, transition, bw="worst",
                                                                         printResults=False)
                print("LME search found a new solution", solnchange, "times and in the first try", both + 1,
                      "LME objs give a solution")
                if FP_objval > 0:
                    # then the simulation
                    for sss in range(sims):
                        timeHorz = T  # if jobarriveprob > 1 else 500000 # number of time periods
                        bigK = 1e2  # if jobarriveprob <= 1.91 else 1e3
                        cC = 2 * objective.max()
                        percent = 90  # Last portion
                        # print(keepMid)
                        print("\nsim ", sss, " of ", ind, " instance with mu ", jobarriveprob,  # + 0.1,
                              " and w BigK ", bigK,
                              "\n with previous sim taking ", simend - simstart, " seconds")

                        simstart = time()
                        # mass, empRew, queuesTree = succfailSim(state, timeHorz, wap, jobarriveprob, wsp,
                        #                                        bigK, objective, cC, percent, zeroprior_s, zeroprior_f)
                        mass, empRew, queuesTree = succfailSim_alt(state, timeHorz, wap, jobarriveprob, wsp,
                                                                   bigK, objective, cC, percent, recordEvery=10,
                                                                   optimalOrLocal='local', iii=0, jjj=zeroprior_f)
                        simend = time()
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
        keepResults[ss][3] = jobarriveprob
        print("It has been ", time() - start, " seconds so far, from the start that is")
        np.savetxt("binaryBetaModel_5priors.csv", keepResults, delimiter=",")
        print()
        print(keepResults)
