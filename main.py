# this is for the tree model
# from itertools import product
# from mip import *
# from warnings import warn
from sys import exit
# from copy import deepcopy
# from scipy.stats import dirichlet

from datetime import datetime

from simEnvironments import *
from simEnvironmentsLinear import *
from supportModules import *
from time import sleep

pd.set_option('display.max_columns', None)


# def dirichletTransitionProb(quantiles, alpha):
#     return dirichlet.pdf(quantiles, alpha)


def transitions(fromState):
    if len(fromState) >= 2:
        probs = fromState / sum(fromState)
    else:
        exit("Problem in transitions. Argument here should be 2 or larger.")
    return probs


def vectorOfMultipliers(rngStates_):
    if rngStates_ > 2:
        baseRewardVector = np.arange(1, rngStates_ + 1) / rngStates_  # e.g. (0.2, 0.4, 0.6, 0.8, 1)
    elif rngStates_ == 2:
        baseRewardVector = np.array([0, 1])
    else:
        exit("Problem in vector of multipliers. Argument here should be 2 or larger.")
    return baseRewardVector


def vectorOfChange_rating(listOfStates, rngStates_, maxNumRat, beta_val, l_state, prior_info = None):
    # dim is the number of total eligible states
    dim = len(listOfStates)
    mat = np.zeros((dim, dim))  # A matrix
    rewardBase = vectorOfMultipliers(rngStates_)
    nudge = np.arange(1, rngStates_ + 1)  # np.array([1, 3, 6, 10, 15])
    rewards = np.zeros(dim)
    for ii in range(dim):
        rewards[ii] = np.dot(transitions(listOfStates[ii]), rewardBase)
        # nudge zero just a little to get the sort proper;
        # also add a little more to the earlier states to manage cases where
        # there are states with the same rewards downstream
        rewards[ii] += (ii == 0) * 1e-8 + \
                       (ii > 0) * 1e-10 / np.dot(listOfStates[ii], nudge[:rngStates_]) # sum(listOfStates[ii])


    rewards_ordered = -np.sort(-rewards)
    indices_ordered = np.argsort(-rewards)  # to order states by decreasing rewards
    # a dictionary where the key is the state (e.g., 00000) and value is its index in indices_ordered
    statesAndIndices_ordered = {}
    for ii in range(dim):
        pos = ""
        index = indices_ordered[ii]
        for j in range(rngStates_):
            pos += str(int(listOfStates[index][j]))
            pos += "_" if j < (rngStates_ - 1) else ""
        statesAndIndices_ordered[pos] = ii
        # remove the nudges
        rewards_ordered[ii] -= (index == 0) * 1e-8 + \
                               (index > 0) * 1e-10 / np.dot(listOfStates[index], nudge[:rngStates_]) # sum(listOfStates[index])

    # transitions use state values (and the prior through the rewards)
    # build the A matrix column by column
    for ii in range(dim):
        index = indices_ordered[ii]  # this is the original index of the state in listOfStates
        if sum(listOfStates[index]) == maxNumRat:  # if in a last state, remain
            mat[ii][ii] = beta_val
        else:  # any middle state feeds into rngStates_-many different states
            checkTransitions = 0
            for kk in range(rngStates_):
                pos = ""
                for jj in range(rngStates_):  # take exactly 1 step in each direction one-by-one
                    pos += str(int(listOfStates[index][jj] + (jj == kk)))
                    pos += "_" if jj < (rngStates_ - 1) else ""
                indexOfMove = statesAndIndices_ordered[pos]
                moveProb = transitions(listOfStates[index])[kk]
                mat[indexOfMove][ii] = beta_val * moveProb
                checkTransitions += moveProb
            if checkTransitions <= 1 - 1e-12:
                exit("Problem in the transition probabilities - they do not sum up to 1.")

    # print("Full A matrix\n", mat, end="\n\n")
    if sum(np.abs(sum(mat) - beta_val)) > 1e-12:
        exit("Problem in the full A matrix.")

    # find the position of state l and remove all rows/columns below/right of it, zero out the last row
    pos = ""
    for jj in range(rngStates_):
        pos += str(int(l_state[jj]))
        pos += "_" if jj < (rngStates_ - 1) else ""
    pos_of_l = statesAndIndices_ordered[pos]
    toRemove = pos_of_l + 1
    mat2 = mat[:toRemove, :toRemove]
    mat2[-1, :] = np.zeros(toRemove)
    # print("Amended A matrix\n", mat2, end="\n\n")

    # M matrix from A
    mMat = np.linalg.inv(np.identity(len(mat2)) - mat2)
    # print("(I-A)^(-1)\n", mMat)

    posOfZero = statesAndIndices_ordered["_".join([str(int(prior_info[i])) for i in range(rngStates_)])]
    sumZero = mMat.sum(axis=0)[posOfZero]
    sum_l = mMat.sum(axis=0)[pos_of_l]

    mVector = mMat[:,posOfZero] / sumZero - mMat[:,pos_of_l] / sum_l

    monotone = False
    # sumToNeg = 0
    if all(mVector[:(posOfZero + 1)] >= 0):
        monotone = True
    sumToNeg = sum(mVector[(posOfZero + 1):])

    positiveChangeInReward = False
    change = np.dot(rewards_ordered[:(pos_of_l+1)], mVector)
    if change >= 0:
        positiveChangeInReward = True

    cumsumCheck = False
    if all(np.cumsum(mVector[:pos_of_l]) >= 0):
    # if all(np.cumsum(mVector[:pos_of_l] * rewards_ordered[:pos_of_l]) >= 0):
        cumsumCheck = True

    if not cumsumCheck and monotone:
        print(mVector)  # this shouldn't be printed at all
        exit("This shouldn't be printed at all.")

    # if not cumsumCheck:
    #     print(mVector)
    #     print(rewards_ordered[:len(mVector)])
    #     print("below and excluding zero sums to " + str(sumToNeg))
    #     exit()

    rewardOfNegChangeStateAboveZero = np.zeros(1)
    for el in range(posOfZero):
        if mVector[el] < 0:
            rewardHere = rewards_ordered[el]
            rewardOfNegChangeStateAboveZero = np.append(rewardOfNegChangeStateAboveZero, rewardHere)
    rewardOfNegChangeStateAboveZero = rewardOfNegChangeStateAboveZero[1:]
    # print(statesAndIndices_ordered)
    return {'mVector': mVector, 'colZeroSum': sumZero, 'monotoneM': monotone,
            'posOfZero': posOfZero, 'positiveChangeInReward': positiveChangeInReward,
            'rewards_ordered': rewards_ordered, 'changeInReward': change,
            'sumBelowZero': sumToNeg, 'mMat': mMat, 'truncatedA': mat2, 'changeInZero': mVector[posOfZero],
            'cumulativeSumsPositive': cumsumCheck, 'rewardOfNegChangeStateAboveZero': rewardOfNegChangeStateAboveZero}




def vectorOfChange_noInverse(listOfStates, rngStates_, maxNumRat, beta_val, l_state, prior_info, changeZero=1):
    # will need to be called with each different state l because flows are based on state zero and state l
    # dim is the number of total eligible states
    dim = len(listOfStates)
    rewardBase = vectorOfMultipliers(rngStates_)
    # get the rewards in
    rewards = np.zeros(dim)
    nudge = np.arange(1, rngStates_ + 1)  # np.array([1,3,6,10,15])
    for ii in range(dim):
        rewards[ii] =  np.dot(transitions(listOfStates[ii]), rewardBase)
        # nudge zero just a little to get the sort proper;
        # also add a little more to the earlier states to manage cases where
        # there are states with the same rewards downstream
        rewards[ii] += (ii == 0) * 1e-8 + \
                       (ii > 0) * 1e-9 / np.dot(listOfStates[ii], nudge[:rngStates_]) # sum(listOfStates[ii])

    rewards_ordered = -np.sort(-rewards)
    indices_ordered = np.argsort(-rewards)  # to order states by decreasing rewards
    # a dictionary where the key is the state (e.g., 00000) and value is its index in indices_ordered
    statesAndIndices_ordered = {}
    # stores a dictionary of multipliers of mass from state 0 and state l; first col state 0, second col state l
    flows_memoized = {}
    state_l_local_index = dim
    reward_state_l = np.dot(transitions(l_state), rewardBase)
    keepAdding = True
    for ii in range(dim):
        pos = ""
        index = indices_ordered[ii]
        for j in range(rngStates_):
            pos += str(int(listOfStates[index][j]))
            pos += "_" if j < (rngStates_ - 1) else ""
        statesAndIndices_ordered[pos] = ii
        # remove the nudges
        rewards_ordered[ii] -= (index == 0) * 1e-8 + \
                               (index > 0) * 1e-9 / np.dot(listOfStates[index], nudge[:rngStates_]) # sum(listOfStates[index])
        if keepAdding: # rewards_ordered[ii] >= reward_state_l: #or
            # must keep adding upstream states with matching rewards to the l-state
            memoizableStates = deepcopy(statesAndIndices_ordered)
        if index == 0:
            flows_memoized[pos] = np.zeros(2)
            flows_memoized[pos][0] = 1
            memoizableStates = deepcopy(statesAndIndices_ordered)
        if sum(np.abs(listOfStates[index] - l_state)) == 0:
            flows_memoized[pos] = np.zeros(2)
            flows_memoized[pos][1] = 1
            state_l_local_index = ii
            memoizableStates = deepcopy(statesAndIndices_ordered)
        if keepAdding and (rewards_ordered[ii] < reward_state_l):
            keepAdding = False

    # print("first time", flows_memoized)
    # should start the construction of flows from 0, function with memoization
    def getFlows(stateArray, beta_val_, memory):  # returns an array of flow multipliers
        pos_ = ""
        for kk_ in range(len(stateArray)):  # get the string of the state
            pos_ += str(int(stateArray[kk_]))
            pos_ += "_" if kk_ < (len(stateArray) - 1) else ""
        # print("I need", pos_)
        if pos_ in memory:  # if exists, all good
            # print("I have", pos_, "it's", memory[pos_])
            return memory[pos_]
        else: # if not, figure out all states that feeds into this one and use them
            # print("I do not have", pos_, end = " so ")
            # first, list all eligible states and call them all at once
            allEligibleStates = np.zeros((len(stateArray), len(stateArray)))
            eligibleStatesCounter = 0
            pos_comes_from_list = []
            transitionFrom = []
            for kk_ in range(len(stateArray)):
                comes_from = deepcopy(stateArray)
                pos_comes_from = ""
                if stateArray[kk_] > prior_info[kk_]:
                    comes_from[kk_] -= 1

                    for jj_ in range(len(stateArray)):  # get the string of the state
                        pos_comes_from += str(int(comes_from[jj_]))
                        pos_comes_from += "_" if jj_ < (len(stateArray) - 1) else ""
                    # check if it's okay to work with that and go on if so
                    if pos_comes_from in memoizableStates:
                        allEligibleStates[eligibleStatesCounter] = comes_from
                        eligibleStatesCounter += 1
                        pos_comes_from_list.append(pos_comes_from)
                        transitionFrom.append(kk_)

                    # if np.dot(transitions(comes_from), rewardBase) > reward_state_l or \
                    #         (np.dot(transitions(comes_from), rewardBase) == reward_state_l and
                    #         sum(comes_from) <= sum(l_state)):  # this is tricky, I need to eliminate flows from states
                    #     # with the same reward as state l but downstream of state l
                    #     allEligibleStates[eligibleStatesCounter] = comes_from
                    #     eligibleStatesCounter += 1
                    #     for jj_ in range(len(stateArray)):  # get the string of the state
                    #         pos_comes_from += str(int(comes_from[jj_]))
                    #         pos_comes_from += "_" if jj_ < (len(stateArray) - 1) else ""
                    #     pos_comes_from_list.append(pos_comes_from)
                    #     transitionFrom.append(kk_)



            allEligibleStates = allEligibleStates[:eligibleStatesCounter, :]
            # print("I am getting", allEligibleStates)
            trans_prob = np.zeros(len(allEligibleStates))
            for kk_ in range(len(allEligibleStates)):
                trans_prob[kk_] = transitions(allEligibleStates[kk_])[int(transitionFrom[kk_])]
            trans_prob /= (1 - beta_val_) if sum(stateArray) == maxNumRat else 1
            # print("probabilities",trans_prob)
            for ijk in range(eligibleStatesCounter):
                memory[pos_comes_from_list[ijk]] = getFlows(allEligibleStates[ijk], beta_val_, memory)
            someList = [trans_prob[ijk] * memory[pos_comes_from_list[ijk]] for
                                    ijk in range(eligibleStatesCounter)]

            return beta_val_ * sum(someList)

    # get all the states with flows from zero and l
    # this must be linear in the number of states
    for ii in range(dim):
        pos = ""
        index = indices_ordered[ii]
        if (rewards_ordered[ii] >= reward_state_l) or (ii <= state_l_local_index):
            # on 5.19.2023, eren changed above line from rewards_ordered[ii] >= reward_state_l to what is now up there
        # if (rewards_ordered[ii] > reward_state_l) or ((rewards_ordered[ii] >= reward_state_l)
        #                                               and (sum(listOfStates[index]) < sum(l_state))):
            for j in range(rngStates_):
                pos += str(int(listOfStates[index][j]))
                pos += "_" if j < (rngStates_ - 1) else ""
            # if pos in flows_memoized:
            #     print("no need to call for", pos, "; it is in my current list, which is", flows_memoized)
            # else:
            #     print("calling for", pos, "; my current list", flows_memoized)
            #     flows_memoized[pos] = getFlows(listOfStates[index], beta_val, flows_memoized)
            if pos not in flows_memoized:
                flows_memoized[pos] = getFlows(listOfStates[index], beta_val, flows_memoized)

            # print('original flow of', pos, ':', flows_memoized[pos])
    # once again loop over all eligible states, i.e., no states strictly worse than state l,
    # to obtain the relationship between state l and state 0 flows
    # the total change must sum up to zero, hence having fixed the change in state 0 to 1,
    # the change in state l is characterized fully - although it requires getting the ratio of
    # sum of the flow multipliers from state 0 (plus one) to sum of the flow multipliers from state l (plus one)
    # negative of this ratio multiplied by the change in state 0 (1) gives the change in state l, which must be negative

    # first, find the numerator and denominator
    numerator = 0
    denominator = 0
    for ii in (ij for ij in range(dim) if ij <= state_l_local_index):
        pos = ""
        index = indices_ordered[ii]

        for j in range(rngStates_):
            pos += str(int(listOfStates[index][j]))
            pos += "_" if j < (rngStates_ - 1) else ""
        if pos in memoizableStates:
            numerator += flows_memoized[pos][0]
            denominator += flows_memoized[pos][1]
    # print("numerator", numerator, "denominator", denominator)
    if denominator == 0:
        print("The denominator is zero. That's a problem.")
    if numerator == 0:
        print("The numerator is zero. That's a problem.")

    # then, modify the flows
    modified_flows = deepcopy(flows_memoized)
    change_state_l = -changeZero * numerator / denominator # if sum(l_state) < maxNumRat else -1
    # print("change_state_l", change_state_l)
    for ii in (ij for ij in range(dim) if ij <= state_l_local_index):
        pos = ""
        index = indices_ordered[ii]
        if ii < state_l_local_index:
            for j in range(rngStates_):
                pos += str(int(listOfStates[index][j]))
                pos += "_" if j < (rngStates_ - 1) else ""
            modified_flows[pos][1] *= change_state_l if modified_flows[pos][1] > 0 else 1
            modified_flows[pos][0] *= changeZero
        else:
            for j in range(rngStates_):
                pos += str(int(listOfStates[index][j]))
                pos += "_" if j < (rngStates_ - 1) else ""
            modified_flows[pos][1] = change_state_l
        # print("modified flow for",pos,"is",modified_flows[pos])


    # below here, I assume I fixed the change in state 0 to 1 (one) unit, and so all the first multipliers of each state
    # has the inflow from state 0
    # I can use rewards_ordered here with ii to figure out that particular state's contribution to the reward
    rewardChange = 0
    for ii in (ij for ij in range(dim) if ij <= state_l_local_index):
        pos = ""
        index = indices_ordered[ii]
        for j in range(rngStates_):
            pos += str(int(listOfStates[index][j]))
            pos += "_" if j < (rngStates_ - 1) else ""
        # print("pos", pos, "flow", sum(modified_flows[pos]))
        rewardChange += sum(modified_flows[pos]) * rewards_ordered[ii]
    # print("indices ordered", indices_ordered)
    return {'modifiedFlows': modified_flows, 'rewardChange': rewardChange, 'memoizableStates': memoizableStates}


def vectorOfChange_succFail(rewards, transitions_, beta_val, l_state):
    dim = int(len(rewards) * (len(rewards) + 1) / 2)  # instance dimension
    mat = np.zeros((dim, dim))  # M matrix, (I-A)^(-1)

    instance2rowcol = {}  # need to transform 2-dim states to a vector of states
    # associate each state with a number, (0,0) is 0, (0,1) is 1, (0,2) is 2, and so on
    # numbering is like this because .ravel() unravels a matrix in this order
    counter_ = -1
    for row in range(len(rewards)):
        for col in range(len(rewards)):
            if row + col < len(rewards):
                counter_ += 1
                instance2rowcol[str(row) + "_" + str(col)] = counter_

    counter_ = -1
    rewards_ordered = -np.sort(-rewards.ravel())
    indices_ordered = np.argsort(-rewards.ravel())  # to order states by decreasing rewards
    flattenedPositions2index = {}  # re-number the states in order of decreasing rewards
    for i in range(dim):
        flattened_index = indices_ordered[i]  # map indices back to 2-dim states
        row = int(np.floor(flattened_index / len(rewards)))
        col = int(flattened_index - row * len(rewards))

        # associate the original numbering of states to ordering by decreasing rewards
        flattened_position = instance2rowcol[str(row) + "_" + str(col)]
        counter_ += 1
        flattenedPositions2index[str(flattened_position)] = counter_


    # build the A matrix column by column
    counter_ = -1
    for i in range(dim):
        flattened_index = indices_ordered[i]
        row = int(np.floor(flattened_index / len(rewards)))
        col = int(flattened_index - row * len(rewards))
        counter_ += 1

        if transitions_[str(row) + "_" + str(col)]['s'] > 0:
            feeds_into = instance2rowcol[str(row + 1) + "_" + str(col)]
            row2write = flattenedPositions2index[str(feeds_into)]
            mat[row2write][counter_] = beta_val * transitions_[str(row) + "_" + str(col)]['s']
        if transitions_[str(row) + "_" + str(col)]['f'] > 0:
            feeds_into = instance2rowcol[str(row) + "_" + str(col + 1)]
            row2write = flattenedPositions2index[str(feeds_into)]
            mat[row2write][counter_] = beta_val * transitions_[str(row) + "_" + str(col)]['f']
        if transitions_[str(row) + "_" + str(col)]['r'] > 0:
            mat[counter_][counter_] = beta_val * transitions_[str(row) + "_" + str(col)]['r']

    # find the number of state l
    flattened_position_l = instance2rowcol[str(int(l_state[0])) + "_" + str(int(l_state[1]))]
    toRemove = flattenedPositions2index[str(flattened_position_l)] + 1
    # will remove every state below it

    # print(rewards)
    # print("Indices ordered w.r.t. rewards", indices_ordered)
    # print("succ, fail to indices", flattenedPositions2index)
    # print("Full A matrix \n", mat, end="\n\n")

    # remove every state below state l
    # print(mat.sum(axis=0))
    mat2 = mat[:toRemove, :toRemove]
    mat2[-1, :] = np.zeros(toRemove)  # last row should be all zeros
    # print("Amended A matrix \n", mat2, end="\n\n")

    # M matrix from A
    mMat = np.linalg.inv(np.identity(len(mat2)) - mat2)
    # print("(I-A)^(-1)\n", mMat)

    # mVector = np.zeros(len(mMat))
    posOfZero = flattenedPositions2index['0']
    pos_of_l = flattenedPositions2index[str(flattened_position_l)]

    sumZero = mMat.sum(axis=0)[posOfZero]
    sum_l = mMat.sum(axis=0)[pos_of_l]

    mVector = mMat[:,posOfZero] / sumZero - mMat[:,pos_of_l] / sum_l

    monotone = False
    # sumToNeg = 0
    if all(mVector[:(posOfZero + 1)] >= 0):
        monotone = True
    # elif sum(mVector[(posOfZero + 1):]) < 0:
    sumToNeg = sum(mVector[(posOfZero + 1):])

    positiveChangeInReward = False
    change = np.dot(rewards_ordered[:(pos_of_l+1)], mVector)
    if change >= 0:
        positiveChangeInReward = True

    cumsumCheck = False
    # if all(np.cumsum(mVector[:pos_of_l]) >= 0):
    if all(np.cumsum(mVector[:pos_of_l] * rewards_ordered[:pos_of_l]) >= 0):
        cumsumCheck = True

    if not cumsumCheck and monotone:
        print(mVector)  # this shouldn't be printed at all

    # if not cumsumCheck:
    #     print(mVector)
    #     print(rewards_ordered[:len(mVector)])
    #     print("below and excluding zero sums to " + str(sumToNeg))
    #     exit()

    return mVector, sumZero, monotone, posOfZero, positiveChangeInReward, \
        rewards_ordered, change, sumToNeg, mMat, mat2, \
        cumsumCheck


def mainSim(beta, numTrans, rngStates, numPriors, size=1e5, fullModel=True):
    all_start = time()
    ratingsList = vectorOfMultipliers(rngStates)  # reward of a state is dot product of this and the state
    eligibleStates = np.zeros((int(size), rngStates))

    if rngStates == 2:
        priorSuccess = numPriors
        priorFail = numPriors
        priorList = np.zeros((int(priorSuccess * priorFail), rngStates))
        counter = 0
        for (m, n) in ((i, j) for (i, j) in product(range(priorFail), range(priorSuccess))):
            # if excludeSome:
            #     if (m + 1 > 3) or ((m + 1 == 3) and (m + 1 + n + 1 >= 8)):
            #         priorList[counter] = np.array([m + 1, n + 1])
            #         counter += 1
            # else:
            priorList[counter] = np.array([m + 1, n + 1])
            counter += 1
        priorList = priorList[:counter]
        print('There are ' + str(counter) + ' priors.')
    elif rngStates > 2:
        if numPriors > 5 or numTrans > 6:
            input("numPriors is " + str(numPriors) + " and numTrans is " + str(numTrans) +
                  ". Are you sure to run this?? Press <ENTER> to continue\n")
        priorList = np.ones((np.power(numPriors + 1, rngStates), rngStates))
        range_list = [numPriors + 1] * rngStates
        counter = 0
        for i in product(*map(range, range_list)):
            candidate = np.array(i)
            if not any(candidate == 0):  # cannot have any zeros because it implies no transition to a similar up state
                priorList[counter] = candidate
                counter += 1
        priorList = priorList[:counter]
        # print(priorList, len(priorList))
        # exit()
        input('There are ' + str(counter) + ' priors. Press <ENTER> to continue\n')
    else:
        exit("Need at least 2 for rngStates.")

    priorBasedOutcomes = np.ones((len(priorList), 2))  # first column for monotonicity, second for cumulative sum
                                                    # remains 1 if all is good for one prior
    worksOut = {}
    # priorList = np.array([[3, 4], [3, 4]])
    # priorList = np.array([[1,1,1,1,26],[1,1,1,2,28]])
    # priorList = np.array([[1, 1, 3, 1, 1]])
    countPriors = 0
    for prInd in range(len(priorList)):
        worseThanZero = 0
        prior = priorList[prInd]  # np.array([1 + (prInd == 0), 1 + (prInd == 1), 1 + (prInd == 2), 1 + (prInd == 3), 1 + (prInd == 4)])
        ratingOfZero = np.dot(transitions(prior), ratingsList)  # prior is always the state zero
        print("\nState zero", prior ,",Rating of zero" , str(ratingOfZero)[:10])
        iter_ = 0
        range_list = [numTrans + 1] * rngStates
        for i in (i for i in product(*map(range, range_list)) if sum(np.array(i)) <= numTrans):
            eligibleStates[iter_, :] = np.array(i) + prior
            if np.dot(transitions(eligibleStates[iter_]), ratingsList) < ratingOfZero - 1e-15:
                worseThanZero += 1
            iter_ += 1
        eligibleStates = eligibleStates[:iter_, :]
        print('There are ' + str(iter_) + ' eligible states and ' + str(worseThanZero) + ' states worse than zero.')
        print(eligibleStates, "\ntotal eligible state", iter_, "\nstates worse than 0:", worseThanZero, "\n\n")

        # state_l = np.zeros(rngStates)
        for i in range(1, len(eligibleStates)):  # skip state zero
            if np.dot(transitions(eligibleStates[i]), ratingsList) < ratingOfZero - 1e-15:
                state_l = eligibleStates[i] ## np.array([4,5])  #
                worseThanZero -= 1
                print("State l is", state_l, ", its rating", np.dot(transitions(eligibleStates[i]), ratingsList))
                if fullModel:
                    res = vectorOfChange_rating(eligibleStates, rngStates, numTrans + sum(prior), beta, state_l, prior)
                    resNoInv = vectorOfChange_noInverse(eligibleStates, rngStates, numTrans + sum(prior), beta, state_l,
                                                        prior, changeZero=res['changeInZero'])
                    # print("Reward change from the inverse thing", res['changeInReward'], "change in zero", res['changeInZero'])
                    # print("Reward change without inverting is", resNoInv['rewardChange'], "difference is",
                    #       round(res['changeInReward']-resNoInv['rewardChange'], 20))
                else:
                    resNoInv = vectorOfChange_noInverse(eligibleStates, rngStates, numTrans + sum(prior), beta, state_l,
                                                        prior)
                    print("Reward change without inverting is", resNoInv['rewardChange'])
                    if resNoInv['rewardChange'] <= 1e-14:
                        print("Modified flows are")
                        print(resNoInv['modifiedFlows'])
                        print("memoizableStates")
                        print(resNoInv['memoizableStates'])
                        exit("Things didn't work.")
                # print("Modified flows are")
                # print(resNoInv['modifiedFlows'])
                # exit()
                # if rngStates == 2 and len(priorList) < 30:
                #     print("M vector", res['mVector'])
                # if resNoInv['rewardChange'] < 1e-10:
                if fullModel:
                    if np.abs(res['changeInReward']-resNoInv['rewardChange']) > 1e-10:
                        print("Modified flows are")
                        print(resNoInv['modifiedFlows'])
                        print("M vector", res['mVector'])
                        exit("Things didn't work.")
                    if res['sumBelowZero'] > 0:
                        exit("below zero sums to " + str(res['sumBelowZero'])[:10])
                    if not res['monotoneM']:
                        priorBasedOutcomes[countPriors, 0] = 0 # monotonicity
                    if not res['cumulativeSumsPositive']:
                        priorBasedOutcomes[countPriors, 1] = 0 # cumulative sums
                    if res['changeInReward'] < 0 or not res['cumulativeSumsPositive']:
                        print("Change in reward", str(res['changeInReward'])[:10], ", cumulative sums?",
                              res['cumulativeSumsPositive'], ", below zero sums to", res['sumBelowZero'],
                              ", col zero sums to", res['colZeroSum'])
                        print("M is monotone?", res['monotoneM'])
                        if res['changeInReward'] < 0:
                            print("Zero at", res['posOfZero'], ", col zero sums to", res['colZeroSum'], ",below zero to",
                              res['sumBelowZero'], "\nM vector", res['mVector'])
                            print("rewards_ordered\n", res['rewards_ordered'])
                            exit("Negative change in reward.")

                        # print("Zero at", res['posOfZero'], ", col zero sum", str(res['colZeroSum'])[:6],
                        #       "below zero", str(res['sumBelowZero'])[:7], "\nM vector dim", len(res['mVector']))
                        # print(res['mVector'])
                        # print("Rewards of states with negative change", res['rewardOfNegChangeStateAboveZero'],
                        #       "\nRating of zero" , str(ratingOfZero)[:10], "\n")
        if worseThanZero != 0:
            print("PROBLEM! Didn't loop over all possible l states.")
            exit()
        monotoneIf = False
        if priorBasedOutcomes[countPriors, 0] == 1:
            worksOut[str(prior)] = 'monotone'
            print("MONOTONE!!!")
            # sleep(0.6)
            monotoneIf = True
        if priorBasedOutcomes[countPriors, 1] == 1 and monotoneIf:
            worksOut[str(prior)] += ' & positiveCumulativeSums'
        elif priorBasedOutcomes[countPriors, 1] == 1:
            worksOut[str(prior)] = 'positiveCumulativeSums'

        countPriors += 1
        # input('Press <ENTER> to continue\n')
    if fullModel:
        originalOps = np.get_printoptions()
        np.set_printoptions(threshold=np.inf)
        print(priorBasedOutcomes)
        np.set_printoptions(**originalOps)
        print(sum(priorBasedOutcomes))
        print(sum(priorBasedOutcomes)/len(priorList))
        print(worksOut)
    print("Successfully finished. Took " + str(time() - all_start)[:6] + " seconds.")


def tree():
    # tree
    all_start = time()
    rng = 3
    start = 4
    res = np.zeros(rng)
    res_cumsum = np.zeros(rng)
    numIns = np.zeros(rng)
    succRatio = np.zeros(rng)
    succRatio_cumsum = np.zeros(rng)
    succ = np.zeros(rng)
    negRatio = np.zeros(rng)
    negChange_obj = np.zeros(rng)
    fails = np.zeros(rng)
    fails_cumsumNotAllPos = np.zeros(rng)
    success_monotonicity = np.zeros(rng)
    fails_cumsumNotAllPos_priors = {}
    success_monotonicity_priors = {}
    priorSuccess = start + rng - 2  # 10
    priorFail = start + rng - 2  # 10
    beta = 0.8
    overallNegChange = 0
    totInstances = 0
    print("Will do instances " + str(start - 1) + " to " + str(start + rng - 2), end=". ")
    print("Ranging priors from (1,1) to (" + str(priorSuccess) + "," + str(priorFail) + ").")
    for st in range(rng):
        start_time = time()
        state = start + st
        fails_cumsumNotAllPos_priors[str(state - 1)] = np.zeros((max(priorSuccess, priorFail) + 1, max(priorSuccess, priorFail)))
        success_monotonicity_priors[str(state - 1)] = np.zeros((max(priorSuccess, priorFail), max(priorSuccess, priorFail)))
        obj = np.zeros((state, state))
        transition = {}
        countPriors = 0
        allGoodInThisPrior = 0
        issueInThisPrior = 0
        issueInThisPrior_cumsumNotAllPos = 0
        print("=" * 30)
        print("Instance size " + str(state - 1))
        print()
        for (m, n) in ((i, j) for (i, j) in product(range(priorSuccess), range(priorFail))):
            countPriors += 1
            succ = m + 1  # (state - 3)
            fail = n + 1
            # print("Prior is (" + str(succ) + ", " + str(fail) + ").")
            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                obj[i][j] = (succ + i) / (succ + fail + i + j)
                transition[str(i) + "_" + str(j)] = {}
                if i + j < (state - 1):
                    transition[str(i) + "_" + str(j)]['s'] = obj[i][j]  # success
                    transition[str(i) + "_" + str(j)]['f'] = 1 - transition[str(i) + "_" + str(j)]['s']  # failure
                    transition[str(i) + "_" + str(j)]['r'] = 0  # remain
                else:
                    transition[str(i) + "_" + str(j)]['s'] = 0  # success
                    transition[str(i) + "_" + str(j)]['f'] = 0  # failure
                    transition[str(i) + "_" + str(j)]['r'] = 1
                # print(i, j, "", transition[str(i) + str(j)].values())
                if sum(transition[str(i) + "_" + str(j)].values()) != 1:
                    exit("problem in (" + str(i) + ", " + str(j) + ")")

            # print("obj\n", obj)
            # loop over all possible l states
            state_l = np.zeros(2)
            localVar = 0
            localVar_all = 0
            localVar_obj = 0
            numIns_for_l_neg = 0
            numIns_for_cumsumNotAllPos = 0
            numIns_for_allGood = 0
            # numIns_for_l = 0
            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1) and obj[i][j] < obj[0][0]):
                state_l[0], state_l[1] = i, j
                totInstances += 1
                # numIns_for_l += 1

                check = vectorOfChange_succFail(obj, transition, beta, state_l)
                numIns[st] += 1
                localVar_all += 1
                if check[2]:
                    res[st] += 1
                    numIns_for_allGood += 1
                    success_monotonicity_priors[str(state - 1)][int(succ - 1), int(fail - 1)] += 1
                    # print("For instance size " + str(state - 1) + ", the current success stories is at " + str(int(res[st])))
                else:
                    if not check[10]:  # check[10] is true if cumsum of M vector is always positive up until state l
                        fails_cumsumNotAllPos_priors[str(state - 1)][int(succ - 1), int(fail - 1)] += 1
                        # print()
                        # print("=" * 30)
                        # print("Cumsum of M vector is not always positive!!!")
                        # print("Prior (" + str(succ) + ", " + str(fail) + "), instance size " + str(state - 1) +
                        #       ". State zero is at " + str(check[3]) + ". State l " + str(state_l) +
                        #       ". Zero column sum " + str(check[1]) + ". M vector:")
                        # print(check[0])  # M vector
                        # print(check[5][:len(check[0])])  # ordered objectives
                        numIns_for_cumsumNotAllPos = 1
                        res_cumsum[st] += 1
                    if check[7] >= 0:
                        print()
                        print("=" * 30)
                        print("Prior is (" + str(succ) + ", " + str(fail) +
                              "). State zero is at " + str(check[3]) + ". State l " + str(state_l) +
                              ". Zero column sum " + str(check[1]) + ". M vector:")
                        print(check[0])  # M vector
                        print(check[5][:int(len(obj) * (len(obj) + 1) / 2)])  # ordered objectives
                        print("change in obj val " + str(check[6]))
                        print(check[8][:, check[3]])
                        print("Zero column of M matrix above.\nPROBLEMATICAL, below (and excluding) Zero sums to " + str(check[7]) + " == PROBLEMATICAL")
                    localVar += 1
                    if not check[4]:
                        negChange_obj[st] += 1
                        localVar_obj += 1
                        print()
                        print("=" * 30)
                        print("Prior is (" + str(succ) + ", " + str(fail) + ") State zero is at " + str(check[3]) +
                              ". State l " + str(state_l) +
                              ". Zero column sum " + str(check[1]) + ". M vector:")
                        print(check[0])  # M vector
                        print(check[5][:int(len(obj) * (len(obj) + 1) / 2)])  # ordered objectives
                        print(check[8][:, check[3]])
                        print("Zero column of M matrix above.\nNegative change in obj " + str(check[6]) +
                              ", below Zero sums to " + str(check[7]) + "\n\n")
                        numIns_for_l_neg = 1
            success_monotonicity_priors[str(state - 1)][int(succ - 1), int(fail - 1)] /= localVar_all
            issueInThisPrior += 1 if numIns_for_l_neg == 1 else 0
            issueInThisPrior_cumsumNotAllPos += 1 if numIns_for_cumsumNotAllPos == 1 else 0
            allGoodInThisPrior += 1 if numIns_for_allGood == localVar_all else 0
            if localVar_obj > 0:
                print("Prior (" + str(succ) + ", " + str(fail) + "), instance size " + str(state - 1) +
                      ", negative change " + str(localVar_obj) + " times.")
            # if localVar == 0:
            #     warn("\n\nPrior (" + str(succ) + ", " + str(fail) + "), instance size " + str(state - 1) + ", there are " + str(localVar) + " issues out of " +
            #           str(localVar_all) + ". Success ratio of " + str((localVar_all - localVar) / localVar_all) + ".\n")
            # else:
            #     print("Prior (" + str(succ) + ", " + str(fail) + "), instance size " + str(state - 1) +
            #           ", negative change " + str(localVar_obj) + " times.")
            #     print("there are " + str(localVar) + " issues out of " +
            #           str(localVar_all) + ". Success ratio of " + str((localVar_all - localVar) / localVar_all) + ".")
        fails_cumsumNotAllPos_priors[str(state - 1)][int(max(priorSuccess, priorFail)), :] = numIns[st]

        overallNegChange += negChange_obj[st]
        fails[st] = issueInThisPrior / countPriors
        fails_cumsumNotAllPos[st] = issueInThisPrior_cumsumNotAllPos / countPriors
        succRatio[st] = res[st] / numIns[st]
        success_monotonicity[st] = allGoodInThisPrior / countPriors
        succRatio_cumsum[st] = res_cumsum[st] / numIns[st]

        pd.DataFrame(fails_cumsumNotAllPos_priors[str(state - 1)]).to_csv("priors/insSize" + str(state - 1) + "_cumsumFails.csv",
                                                                          index=False, header=False)

        pd.DataFrame(success_monotonicity_priors[str(state - 1)]).to_csv("priors/insSize" + str(state - 1) + "_monotonicitySuccesses.csv",
                                                                          index=False, header=False)
        # succ[st] = allGoodInThisPrior / countPriors
        # negRatio[st] = negChange_obj[st] / numIns[st]
        print(str(countPriors) + " priors in total in this instance size. " + str(int(numIns[st])) +
              " checked l states.\n" + "Took " + str(time() - start_time)[:6] + " seconds. Success ratio " +
              str(success_monotonicity[st])[:6] + ". Across alll instances " + str(succRatio[st])[:6] + ".\n" +
              str(1 - fails_cumsumNotAllPos[st])[:6] + " of instances where cumulative sums are all positive.")

        # print("Success ratio for instance size " + str(state - 1) + " is " + str(succRatio[st]))
        # input('Press <ENTER> to continue\n')
    print()
    print("="*30, "\n")
    print("Instance sizes " + str(start - 1 + np.arange(rng)))
    print("Success ratios (monotonicity in M vector) across all instances")
    print(succRatio)
    print("Success ratios (monotonicity in M vector) across all priors - fail if a state l fails monotonicity")
    print(success_monotonicity)
    print("Objective value decreased " + str(overallNegChange) + " times, out of " +str(totInstances) + " instances.")
    print("Failure ratios (decrease in rewards) for at least one l in each prior ")
    print(fails) #negRatio
    print("Success ratios for cumulative sums being all positive for all l in each prior ")
    print(1 - fails_cumsumNotAllPos)
    print("Over all instances")
    print(1 - succRatio_cumsum)
    # for st in range(rng):
    #     state = start + st
        # print(fails_cumsumNotAllPos_priors[str(state - 1)])
    print("Took " + str(time() - all_start)[:6] + " seconds.")
    # raise SystemExit(0)
    # feedback
    # beta = 0.5
    # mu_ = 1.5
    # lambda_ = 1.5
    # obj = np.array([0.11, 0.1, 1])
    # transitions = np.array([[0, 1, 0],
    #                         [0, 0, 1],
    #                         [0, 0, 1]])  # transition is in rows, first column first row: going from 0 to 0
    # second column first row: going from 0 to 1
    # third column first row: going from 0 to 2

    # soln_sub, obj_, capacity = feedbackOpt(n=len(obj), lambd=lambda_, mu=mu_, prevSoln=np.empty(0), usePrevSoln=False,
    #                                        transitions=transitions, rewardMult=obj, beta=beta)
    # print()
    # feedbackFixedPoint(n=len(obj), lambd=lambda_, mu=min(mu_, capacity), transition=transitions, rewardMult=obj,
    #                    beta=beta)

    # tree

    def m_Matrix(rewards, transitions_, beta_, state_l_):
        dim = int(len(rewards) * (len(rewards) + 1) / 2)
        mat = np.zeros((dim, dim))

        succfail2rowcol = {}
        counter_ = -1
        for i in range(len(rewards)):
            for j in range(len(rewards)):
                if i + j < len(rewards):
                    counter_ += 1
                    succfail2rowcol[str(i) + "_" + str(j)] = counter_

        counter_ = -1
        indices_ordered = np.argsort(-rewards.ravel())
        flattenedPositions2index = {}
        for i in range(dim):
            flattened_index = indices_ordered[i]
            row = int(np.floor(flattened_index / len(rewards)))
            col = int(flattened_index - row * len(rewards))
            flattened_position = succfail2rowcol[str(row) + "_" + str(col)]
            counter_ += 1
            flattenedPositions2index[str(flattened_position)] = counter_
            # print(flattened_position, row, col)

        counter_ = -1
        for i in range(dim):
            flattened_index = indices_ordered[i]
            row = int(np.floor(flattened_index / len(rewards)))
            col = int(flattened_index - row * len(rewards))
            counter_ += 1
            # flattened_position = succfail2rowcol[str(row) + str(col)]
            # print(flattened_position, row, col)
            if transitions_[str(row) + "_" + str(col)]['s'] > 0:
                feeds_into = succfail2rowcol[str(row + 1) + "_" + str(col)]
                row2write = flattenedPositions2index[str(feeds_into)]
                mat[row2write][counter_] = beta_ * transitions_[str(row) + "_" + str(col)]['s']
            if transitions_[str(row) + "_" + str(col)]['f'] > 0:
                feeds_into = succfail2rowcol[str(row) + "_" + str(col + 1)]
                row2write = flattenedPositions2index[str(feeds_into)]
                mat[row2write][counter_] = beta_ * transitions_[str(row) + "_" + str(col)]['f']
            if transitions_[str(row) + "_" + str(col)]['r'] > 0:
                mat[counter_][counter_] = beta_ * transitions_[str(row) + "_" + str(col)]['r']

        flattened_position = succfail2rowcol[str(int(state_l_[0])) + "_" + str(int(state_l_[1]))]
        toRemove = flattenedPositions2index[str(flattened_position)] + 1
        print(rewards)
        print("Indices ordered w.r.t. rewards", indices_ordered)
        print("succ, fail to indices", flattenedPositions2index)
        print("Full A matrix \n", mat, end="\n\n")
        mat2 = mat[:toRemove, :toRemove]
        mat2[-1, :] = np.zeros(toRemove)

        # build the transitions matrix which will abide by the indices of the flattened matrix
        # succfail2rowcol = {}
        # counter = -1
        # for i in range(len(rewards)):
        #     for j in range(len(rewards)):
        #         if i + j < len(rewards):
        #             counter += 1
        #             succfail2rowcol[str(i) + str(j)] = counter
        #             if transitions[str(i) + str(j)]['s'] > 0:
        #                 mat[counter + len(rewards) - i][counter] = beta * transitions[str(i) + str(j)]['s']
        #             if transitions[str(i) + str(j)]['f'] > 0:
        #                 mat[counter + 1][counter] = beta * transitions[str(i) + str(j)]['f']
        #             if transitions[str(i) + str(j)]['r'] > 0:
        #                 mat[counter][counter] = beta * transitions[str(i) + str(j)]['r']
        #
        #
        # indices_ordered = np.argsort(-rewards.ravel())
        # print(indices_ordered)
        # moved = 0
        # for i in range(dim):
        #     flattened_index = indices_ordered[i]
        #     row = int(np.floor(flattened_index / len(rewards)))
        #     col = int(flattened_index - row * len(rewards))
        #     flattened_position = succfail2rowcol[str(row) + str(col)]
        #     print(flattened_position)
        #
        #
        #
        #     moved += 1

        # if row == state_l[0] and col == state_l[1]:
        #     break

        # build the transitions matrix which will abide by the indices of the flattened matrix
        # for (i, j) in ((i, j) for (i, j) in product(range(len(rewards)), range(len(rewards))) if i + j < len(rewards)):
        #     if transitions[str(i) + str(j)]['s'] > 0:
        #         mat[i + len(rewards)][i * len(rewards) + j] = beta * transitions[str(i) + str(j)]['s']
        #     if transitions[str(i) + str(j)]['f'] > 0:
        #         mat[i + 1][i * len(rewards) + j] = beta * transitions[str(i) + str(j)]['f']
        #     if transitions[str(i) + str(j)]['r'] > 0:
        #         ij = i * (len(rewards) - 1) + j
        #         mat[ij][ij] = beta * transitions[str(i) + str(j)]['r']

        return mat2, flattenedPositions2index, indices_ordered, toRemove


    state = 7  # 11
    beta = 0.5 # 0.75
    mu_ = 0.85 # 0.5
    lambda_ = 0.50 # 0.3
    Delta = 0.005
    obj = np.zeros((state, state))
    transition = {}
    trying = 1
    smallestRatio = 1
    # eps = 0.01

    for (m_, n_) in ((i, j) for (i, j) in product(range(trying), range(trying))):
        succ = n_ + 1  # (state - 3)
        fail = m_ + 1
        print("base state success", succ, "failure", fail)
        for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
            obj[i][j] = (succ + i) / (succ + fail + i + j)
            transition[str(i) + str(j)] = {}
            if i + j < (state - 1):
                transition[str(i) + str(j)]['s'] = obj[i][j]  # success
                transition[str(i) + str(j)]['f'] = 1 - transition[str(i) + str(j)]['s']  # failure
                transition[str(i) + str(j)]['r'] = 0  # remain
            else:
                transition[str(i) + str(j)]['s'] = 0  # success
                transition[str(i) + str(j)]['f'] = 0  # failure
                transition[str(i) + str(j)]['r'] = 1
            print(i, j, "", transition[str(i) + str(j)].values())
            if sum(transition[str(i) + str(j)].values()) != 1:
                exit("problem in (" + str(i) + ", " + str(j) + ")")
        slacks = np.ones((state, state)) * max(mu_, lambda_ / (1 - beta))
        # print("slacks\n", slacks, "\n", "-" * 20)
        print("obj\n", obj)

        newSoln, _, objVal, mass = succfailOpt(n=len(obj), beta=beta, lambd=lambda_, mu=mu_,
                                                 prevSoln=np.empty(0), usePrevSoln=False,
                                                 objective=obj, tr=transition, slacks=slacks)
        LME_soln, objVal_LME, _, _ = succfailFixedPoint(n=len(obj), beta=beta, lambd=lambda_,
                                                 mu=min(mu_, mass), objective=obj, tr=transition)

        LME_soln2, objVal_LME2, _, _ = succfailFixedPoint(n=len(obj), beta=beta, lambd=lambda_+Delta,
                                                        mu=min(mu_, mass), objective=obj, tr=transition)

        state_l = np.zeros(2)
        state_l2 = np.zeros(2)
        minRewUsedState = np.max(obj)
        minRewUsedState2 = np.max(obj)

        for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
            if obj[i][j] <= minRewUsedState and LME_soln[i][j] > 0:
                minRewUsedState = obj[i][j]
                state_l[0], state_l[1] = i, j
        for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
            if obj[i][j] <= minRewUsedState2 and LME_soln2[i][j] > 0:
                minRewUsedState2 = obj[i][j]
                state_l2[0], state_l2[1] = i, j

        if state_l[0] != state_l2[0] or state_l[1] != state_l2[1]:
            exit("problem for the prior success " + str(succ) + ", failure " + str(fail))


        # if smallestRatio > objVal_LME / objVal:
        #     smallestRatio = objVal_LME / objVal
        # print("Ratio of lme/opt", objVal_LME / objVal)
        # print("\nOpt solution")
        # print(newSoln)

        print("state_l", state_l, "\n")
        A_matrix, posOfStates, indicesOrdered, posOf_l = m_Matrix(obj, transition, beta, state_l)
        print("state 0:", posOfStates['0'], "\n")
        print("A_matrix\n", A_matrix)
        print()
        I_A_inverse = np.linalg.inv(np.identity(len(A_matrix)) - A_matrix)
        print("(I-A)^(-1)\n", I_A_inverse)
        # print("check the difference in the last column and the zero's column", end=" ")
        # print(I_A_inverse[:,posOfStates['0']] - I_A_inverse[:,-1])
        cVector = np.zeros(posOf_l)
        cVector[posOf_l - 1] = LME_soln2[int(state_l[0])][int(state_l[1])] - LME_soln[int(state_l[0])][int(state_l[1])]
        cVector[posOfStates['0']] = Delta

        print("cVector", cVector)
        resVec = np.dot(I_A_inverse, cVector)
        print("M matrix dot product with c")
        print(resVec)
        print("base state success", succ, "failure", fail)
        print("increase in LME reward", objVal_LME2 - objVal_LME)
        # sleep(5)
        print()
        check = vectorOfChange_succFail(obj, transition, beta, state_l)
        print(check[0]*check[1]*Delta)
        input('Press <ENTER> to continue\n')

        print("=" * 30)
        print("\n\n")
    # print("Smallest ratio of lme/opt", smallestRatio)
    # raise SystemExit(0)


    if state <= 4:
        for (mm, nn) in ((i, j) for (i, j) in product(range(trying), range(trying))):
            aa = mm + 1
            bb = nn + 1
            print("success", aa, "failure", bb)
            obj[0][0] = aa / (aa + bb)
            obj[0][1] = aa / (aa + bb + 1)
            obj[1][0] = min((aa + 1) / (aa + bb + 1) * 1, 1)
            if state == 3:
                obj[1][1] = obj[0][0] + 0.01 # min(obj[0][0] * 2, 1)  #
                obj[0][2] = aa / (aa + bb + 2)
                obj[2][0] = min((aa + 2) / (aa + bb + 2) * 1, 1)
            if state == 4:
                obj[2][2] = obj[1][1] + 0.001 # min(obj[0][0] * 2, 1)  #
                obj[0][3] = aa / (aa + bb + 3)
                obj[3][0] = min((aa + 3) / (aa + bb + 3) * 1, 1)
                obj[1][2] = (aa + 1) / (aa + bb + 3)
                obj[2][1] = min((aa + 2) / (aa + bb + 3) * 1, 1)
            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j < (state - 1)):
                transition[str(i) + str(j)] = {}
                if j == 5:
                    transition[str(i) + str(j)]['s'] = (aa + i) / (aa + bb + i + j)  # success
                    transition[str(i) + str(j)]['f'] = 0  # failure
                    transition[str(i) + str(j)]['r'] = 1 - transition[str(i) + str(j)]['s'] - \
                                                       transition[str(i) + str(j)]['f']  # remain
                elif i == 5:
                    transition[str(i) + str(j)]['s'] = 0  # success
                    transition[str(i) + str(j)]['f'] = (bb + i) / (aa + bb + i + j)  # failure
                    transition[str(i) + str(j)]['r'] = 1 - transition[str(i) + str(j)]['s'] - \
                                                       transition[str(i) + str(j)]['f']  # remain
                else:
                    transition[str(i) + str(j)]['s'] = (aa + i) / (aa + bb + i + j)  # success
                    transition[str(i) + str(j)]['f'] = 1 - transition[str(i) + str(j)]['s']  # failure
                    transition[str(i) + str(j)]['r'] = 0  # remain
                print(i, j, "", transition[str(i) + str(j)].values())
                if sum(transition[str(i) + str(j)].values()) != 1:
                    exit("problem in (" + str(i) + ", " + str(j) + ")")

            slacks = np.ones((state, state)) * max(mu_, lambda_ / (1 - beta))
            # print("slacks\n", slacks, "\n", "-" * 20)
            print("obj\n", obj)
            newSoln, _, objVal, mass = succfailOpt(n=len(obj), beta=beta, lambd=lambda_, mu=mu_,
                                                     prevSoln=np.empty(0), usePrevSoln=False,
                                                     objective=obj, tr=transition, slacks=slacks)
            _, objVal_LME, _, _ = succfailFixedPoint(n=len(obj), beta=beta, lambd=lambda_,
                                                     mu=min(mu_, mass), objective=obj, tr=transition)
            if smallestRatio > objVal_LME/objVal:
                smallestRatio = objVal_LME / objVal
            print("Ratio of lme/opt", objVal_LME/objVal)
            print("\nOpt solution")
            print(newSoln)
            print("=" * 30)
            print("\n\n")
        print("Smallest ratio of lme/opt", smallestRatio)
        raise SystemExit(0)

    if state >= 3:
        obj[0][0] = 0.5  # + .101
        obj[0][1] = 0.4  # 0.35
        obj[0][2] = 0.3  # 0.87233
        obj[1][0] = 0.7
        obj[1][1] = 0.6  # - 0.101  # platform decides to charge a fixed transaction fee?
        # To discourage matches (instead of encouraging lower reward states by giving out incentives)
        # this discouragement pushes OPT to be an LME, w/o any change in the objective value(?)
        obj[2][0] = 0.8

    if state == 4:
        obj[0][0] = 0.5
        obj[0][1] = 0.4
        obj[0][2] = 0.3
        obj[0][3] = 0.2
        obj[1][0] = 0.7
        obj[1][1] = 0.6
        obj[1][2] = 0.25
        obj[2][0] = 0.8
        obj[2][1] = 0.45
        obj[3][0] = 0.9

    if state >= 3:
        transition = {}
        for (i, j) in ((i, j) for (i, j) in product(range(len(obj)), range(len(obj))) if i + j < (len(obj) - 1)):
            transition[str(i) + str(j)] = {}
            transition[str(i) + str(j)]['s'] = (i + 1) / (i + j + 2)  # success
            transition[str(i) + str(j)]['f'] = 1 - transition[str(i) + str(j)]['s']  # failure
            transition[str(i) + str(j)]['r'] = 0  # remain
            print(i, j, "", transition[str(i) + str(j)].values())
            if sum(transition[str(i) + str(j)].values()) != 1:
                exit("problem in (" + str(i) + ", " + str(j) + ")")
        # transition[str(1) + str(0)]['s'] = 1/30  # success
        # transition[str(1) + str(0)]['f'] = 29/30
    slacks = np.ones((state, state)) * max(mu_, lambda_ / (1 - beta))
    # for (i, j) in product(range(len(obj)), range(len(obj))):
    #     slacks[i][j] = 0 if obj[0][0] < obj[i][j] else mu_
    print("slacks\n", slacks, "\n", "-" * 20)
    newSoln, _, objVal, mass = succfailOpt(n=len(obj), beta=beta, lambd=lambda_, mu=mu_,
                                             prevSoln=np.empty(0), usePrevSoln=False,
                                             objective=obj, tr=transition, slacks=slacks)
    _, objVal_LME, _, _ = succfailFixedPoint(n=len(obj), beta=beta, lambd=lambda_,
                                             mu=min(mu_, mass), objective=obj, tr=transition)
    print("Diff in opt vs lme ", objVal - objVal_LME)
    print("\nOpt solution")
    print(newSoln)
    # raise SystemExit(0)


def denemeSimulation():
    start = time()
    numState = 3
    # lambda
    workerArriveProbability =  0.2 # 0.1 #
    # beta
    workerStayProbability =  0.8 # 0.95 #
    # mu
    jobArriveProbability =  0.6 # 0.5 #

    objVals = np.zeros((numState, numState))
    objVals[0][0] = 0.5
    objVals[0][1] = 0.4
    objVals[0][2] = 0.3
    # objVals[0][3] = 0.2
    objVals[1][0] = 0.7
    objVals[1][1] = 0.6
    # objVals[1][2] = 0.5
    objVals[2][0] = 0.8
    # objVals[2][1] = 0.7
    # objVals[3][0] = 0.9
    doPrimalDual = False
    if doPrimalDual:
    # for primal dual
        transition = {}
        for (i, j) in ((i, j) for (i, j) in product(range(numState), range(numState)) if i + j <= (numState - 1)):
            transition[str(i) + "_" + str(j)] = {}
            if i + j < (numState - 1):
                transition[str(i) + "_" + str(j)]['s'] = objVals[i][j]  # success
                transition[str(i) + "_" + str(j)]['f'] = 1 - transition[str(i) + "_" + str(j)]['s']  # failure
                transition[str(i) + "_" + str(j)]['r'] = 0  # remain
            else:
                transition[str(i) + "_" + str(j)]['s'] = 0  # success
                transition[str(i) + "_" + str(j)]['f'] = 0  # failure
                transition[str(i) + "_" + str(j)]['r'] = 1
            # print(i, j, "", transition[str(i) + str(j)].values())
            if sum(transition[str(i) + "_" + str(j)].values()) != 1:
                exit("problem in (" + str(i) + ", " + str(j) + ")")

        succfailOpt(numState, workerStayProbability, workerArriveProbability, jobArriveProbability, np.empty(0), False,
                    objVals, transition, np.ones((numState, numState)) * 2)
        succfailFixedPoint(n=len(objVals), beta=workerStayProbability, lambd=workerArriveProbability,
                           mu=jobArriveProbability, objective=objVals, tr=transition)
        succfailDual(numState, workerStayProbability, workerArriveProbability, jobArriveProbability, objVals)
        exit()
    # for primal dual

    TT = int(1e6)
    bigK = 1000
    cC = 2 * objVals[numState - 1][0]
    recordEvery = 10
    optOrLocalOrBoth = 'both' # 'optimal', 'local', 'both'

    if optOrLocalOrBoth == 'local' or optOrLocalOrBoth == 'both':
        print("Local pricing.")
        track_mass, total_reward, track_queues = succfailSim(numState, TT, workerArriveProbability, jobArriveProbability,
                                                             workerStayProbability, bigK, objVals,
                                                             cC, 80, recordEvery)

        df_massTree = pd.DataFrame(track_mass,
                                   columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                              ((i, j) for (i, j) in
                                                               product(range(numState), range(numState)) if
                                                               i + j <= (numState - 1))]], dtype=float)
        df_qsTree = pd.DataFrame(track_queues,
                                   columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                              ((i, j) for (i, j) in
                                                               product(range(numState), range(numState)) if
                                                               i + j <= (numState - 1))]], dtype=float)

        for (i, j) in ((i, j) for (i, j) in product(range(numState), range(numState)) if i + j <= (numState - 1)):
            name = 'State(' + str(i) + ',' + str(j) + ')'
            namep_just = 'Price(' + str(i) + ',' + str(j) + ')'

            # local price #
            df_qsTree[namep_just] = df_qsTree.apply(lambda x: cC * (bigK - min(bigK, x[name])) / bigK, axis=1)
            # local price #

        # df_qsTree.to_csv("pricesOverTimeTree.csv", index=False)
        # df_massTree.to_csv("massesOverTime.csv", index=False)
        print(df_qsTree)
        print("took " + str(time() - start)[:6])

        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(0,0)'].to_numpy(), color='green', label='State 0,0')
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1,0)'].to_numpy(), color='red', label='State 1,0')
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(2,0)'].to_numpy(), color='blue', label='State 2,0')
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1,1)'].to_numpy(), color='purple', label='State 1,1')
        plt.ylabel('Match Rates')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(0,0)'].to_numpy(), color='green', label='State 0,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(1,0)'].to_numpy(), color='red', label='State 1,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(2,0)'].to_numpy(), color='blue', label='State 2,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(1,1)'].to_numpy(), color='purple', label='State 1,1')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    if optOrLocalOrBoth == 'optimal' or optOrLocalOrBoth == 'both':
        print("Optimal pricing.")
        start = time()
        track_mass, total_reward, track_queues = succfailSim(numState, TT, workerArriveProbability, jobArriveProbability,
                                                             workerStayProbability, bigK, objVals,
                                                             cC, 80, recordEvery, 'optimal')

        df_massTree = pd.DataFrame(track_mass,
                                   columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                              ((i, j) for (i, j) in
                                                               product(range(numState), range(numState)) if
                                                               i + j <= (numState - 1))]], dtype=float)

        df_qsTree = pd.DataFrame(track_queues,
                                 columns=['Time'.split() + ['State(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                            ((i, j) for (i, j) in
                                                             product(range(numState), range(numState)) if
                                                             i + j <= (numState - 1))]], dtype=float)

        for (i, j) in ((i, j) for (i, j) in product(range(numState), range(numState)) if i + j <= (numState - 1)):
            name = 'State(' + str(i) + ',' + str(j) + ')'

            name_s = 'State(' + str(i + 1) + ',' + str(j) + ')'
            name_f = 'State(' + str(i) + ',' + str(j + 1) + ')'
            namep = 'OptPrice(' + str(i) + ',' + str(j) + ')'
            namep_eff = 'EffReward(' + str(i) + ',' + str(j) + ')'
            wsp = workerStayProbability
            if i + j < (numState - 1):
                # if (i + j == 0) or (i * j >= 1):
                df_qsTree[namep] = df_qsTree.apply(
                    lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
                                    wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, x[name_s])) / bigK -
                                    wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, x[name_f])) / bigK), axis=1)
                df_qsTree[namep_eff] = df_qsTree.apply(
                    lambda x: objVals[i][j] - cC * ((bigK - min(bigK, x[name])) / bigK -
                                                    wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, x[name_s])) / bigK -
                                                    wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, x[name_f])) / bigK), axis=1)
                # elif i == 0:
                #     df_qsTree[namep] = df_qsTree.apply(
                #         lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
                #                         wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, x[name_f])) / bigK), axis=1)
                # elif j == 0:
                #     df_qsTree[namep] = df_qsTree.apply(
                #         lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
                #                         wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, x[name_s])) / bigK), axis=1)
            elif i + j == (numState - 1):
                df_qsTree[namep] = df_qsTree.apply(lambda x: cC * ((bigK - min(bigK, x[name])) / bigK -
                                                                   wsp * (bigK - min(bigK, x[name])) / bigK), axis=1)
                df_qsTree[namep_eff] = df_qsTree.apply(lambda x: objVals[i][j] - cC *
                                                                 ((bigK - min(bigK, x[name])) / bigK -
                                                                  wsp * (bigK - min(bigK, x[name])) / bigK), axis=1)

        print(df_qsTree)
        print("took " + str(time() - start)[:6])

        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(0,0)'].to_numpy(), color='green', label='State 0,0')
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1,0)'].to_numpy(), color='red', label='State 1,0')
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(2,0)'].to_numpy(), color='blue', label='State 2,0')
        plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1,1)'].to_numpy(), color='purple', label='State 1,1')
        plt.ylabel('Match Rates')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['OptPrice(0,0)'].to_numpy(), color='green', label='State 0,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['OptPrice(1,0)'].to_numpy(), color='red', label='State 1,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['OptPrice(2,0)'].to_numpy(), color='blue', label='State 2,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['OptPrice(1,1)'].to_numpy(), color='purple', label='State 1,1')
        plt.ylabel('Optimal Price')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['EffReward(0,0)'].to_numpy(), color='green', label='State 0,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['EffReward(1,0)'].to_numpy(), color='red', label='State 1,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['EffReward(2,0)'].to_numpy(), color='blue', label='State 2,0')
        plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['EffReward(1,1)'].to_numpy(), color='purple', label='State 1,1')
        plt.ylabel('Reward Adjusted Price')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

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


def denemeSimulationLinear(TT, optimalOrLocal):
    start = time()
    numState = 3
    workerArriveProbability = 1
    if workerArriveProbability == 1:
        lambdaVal = 'One'
    elif workerArriveProbability == 0.5:
        lambdaVal = 'Half'
    else:
        # lambdaVal = "_" + str(workerArriveProbability) + "_"
        exit("Pick lambda as either 1 or 1/2.")
    workerStayProbability = 0.5
    jobArriveProbability = 1
    eps = 0.6  # 0.05 or 0.8

    objVals = np.zeros(numState)
    initQueues = np.zeros(numState)
    # initQueues[0], initQueues[1], initQueues[2] = 980, 0, 0
    objVals[0] = eps
    objVals[1] = eps / 2
    objVals[2] = 1
    transition = np.zeros((numState, numState))
    transition[0][1] = 1
    transition[1][2] = 1
    transition[2][2] = 1
    print("State transitions \n", transition)


    recordEvery = 10
    bigK = 1000
    if optimalOrLocal == 'local' or optimalOrLocal == 'both':
        track_mass, total_reward, track_queues = succfailLinear(numState, TT, workerArriveProbability, jobArriveProbability,
                                                                workerStayProbability, bigK, objVals, transition, initQueues,
                                                                2 * objVals[numState - 1], 80, recordEvery, 'local')

        df_massTree = pd.DataFrame(track_mass, columns=['Time'.split() + ['State(' + str(i) + ')'
                                                                          for i in range(numState)]], dtype=float)

        # df_massTree.to_csv("massesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        #                    "Pricing.csv", index=False)
        # print(df_massTree['Time'])
        if TT <= 5e7:
            plt.figure(figsize=(7, 5), dpi=100)
            # plt.rc('axes', axisbelow=True)
            plt.grid(lw=1.1)
            plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(0)'].to_numpy(), color='green', label='State 0')
            plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1)'].to_numpy(), color='red', label='State 1')
            plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(2)'].to_numpy(), color='blue', label='State 2')
            plt.ylabel('Match Rate')
            plt.xlabel('Time')
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            # plt.tight_layout()
            plt.title('Local Pricing')
            # plt.savefig("massesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
            #             "Pricing.eps", format='eps', bbox_inches='tight')
            plt.show()
            # plt.cla()

        pd.DataFrame(track_queues,
                     columns=['Time'.split() + ['State(' + str(i) + ')' for i in range(numState)]],
                     dtype=float).to_csv("queuesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
                                         "Pricing.csv", index=False)

        df_qsPrice = pd.DataFrame(track_queues, columns=['Time'.split() + ['State(' + str(i) + ')'
                                                                          for i in range(numState)]], dtype=float)
        df_qsRewAdjPrice = pd.DataFrame(track_queues, columns=['Time'.split() + ['State(' + str(i) + ')'
                                                                          for i in range(numState)]], dtype=float)
        df_qsPriceLocal = deepcopy(df_qsPrice)
        df_qsRewAdjPriceLocal = deepcopy(df_qsRewAdjPrice)

    if optimalOrLocal == 'optimal' or optimalOrLocal == 'both':
        track_mass, total_reward, track_queues = succfailLinear(numState, TT, workerArriveProbability, jobArriveProbability,
                                                                workerStayProbability, bigK, objVals, transition,
                                                                initQueues,
                                                                2 * objVals[numState - 1], 80, recordEvery, 'optimal')

        df_massTree = pd.DataFrame(track_mass, columns=['Time'.split() + ['State(' + str(i) + ')'
                                                                          for i in range(numState)]], dtype=float)

        # df_massTree.to_csv("massesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        #                    "Pricing.csv", index=False)
        # print(df_massTree['Time'])
        if TT <= 5e7:
            plt.figure(figsize=(7, 5), dpi=100)
            # plt.rc('axes', axisbelow=True)
            plt.grid(lw=1.1)
            plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(0)'].to_numpy(), color='green', label='State 0')
            plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1)'].to_numpy(), color='red', label='State 1')
            plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(2)'].to_numpy(), color='blue', label='State 2')
            plt.ylabel('Match Rate')
            plt.xlabel('Time')
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            # plt.tight_layout()
            plt.title('Optimal Pricing')
            # plt.savefig("massesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
            #             "Pricing.eps", format='eps', bbox_inches='tight')
            plt.show()
            # plt.cla()

        pd.DataFrame(track_queues,
                     columns=['Time'.split() + ['State(' + str(i) + ')' for i in range(numState)]],
                     dtype=float).to_csv("queuesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
                                         "Pricing.csv", index=False)

        df_qsPrice = pd.DataFrame(track_queues, columns=['Time'.split() + ['State(' + str(i) + ')'
                                                                           for i in range(numState)]], dtype=float)
        df_qsRewAdjPrice = pd.DataFrame(track_queues, columns=['Time'.split() + ['State(' + str(i) + ')'
                                                                                 for i in range(numState)]], dtype=float)
        df_qsPriceOpt = deepcopy(df_qsPrice)
        df_qsRewAdjPriceOpt = deepcopy(df_qsRewAdjPrice)

    for i in range(numState):
        namep = 'State(' + str(i) + ')'
        if optimalOrLocal == 'local' or optimalOrLocal == 'both':
            # print("Doing local prices.")
            df_qsPriceLocal[namep] = df_qsPriceLocal.apply(lambda x: 2 * objVals[numState - 1] *
                                                        (bigK - min(bigK, x[namep])) / bigK, axis=1)
            df_qsRewAdjPriceLocal[namep] = df_qsRewAdjPriceLocal.apply(lambda x: objVals[i] - 2 * objVals[numState - 1] *
                                                                       (bigK - min(bigK, x[namep])) / bigK, axis=1)
        if optimalOrLocal == 'optimal' or optimalOrLocal == 'both':
            # optimal price
            # print("Doing optimal prices.", end=" ")
            name_feed = 'State(' + str(i + 1) + ')'
            if i < numState - 1:
                # print("For " + namep + " to " + name_feed)
                df_qsPriceOpt[namep] = df_qsPriceOpt.apply(
                    lambda x: 2 * objVals[numState - 1] * ((bigK - min(bigK, x[namep])) / bigK -
                    workerStayProbability * transition[i][i + 1] * (bigK - min(bigK, x[name_feed])) / bigK), axis=1)
                df_qsRewAdjPriceOpt[namep] = df_qsRewAdjPriceOpt.apply(
                    lambda x: objVals[i] - 2 * objVals[numState - 1] * ((bigK - min(bigK, x[namep])) / bigK -
                    workerStayProbability * transition[i][i + 1] * (bigK - min(bigK, x[name_feed])) / bigK), axis=1)
            else:
                # print()
                df_qsPriceOpt[namep] = df_qsPriceOpt.apply(lambda x: 2 * objVals[numState - 1] *
                                                               (bigK - min(bigK, x[namep])) / bigK
                                                               * (1 - workerStayProbability), axis=1)
                df_qsRewAdjPriceOpt[namep] = df_qsRewAdjPriceOpt.apply(lambda x: objVals[i] - 2 * objVals[numState - 1] *
                                                                           (bigK - min(bigK, x[namep])) / bigK
                                                                           * (1 - workerStayProbability), axis=1)

    # df_qsPrice.to_csv("pricesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
    #                   "Pricing.csv", index=False)
    if TT <= 5e7 and (optimalOrLocal == 'local' or optimalOrLocal == 'both'):
        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)
        plt.plot(df_qsPriceLocal['Time'].to_numpy(), df_qsPriceLocal['State(0)'].to_numpy(), color='green',
                 label='State 0')
        plt.plot(df_qsPriceLocal['Time'].to_numpy(), df_qsPriceLocal['State(1)'].to_numpy(), color='red',
                 label='State 1')
        plt.plot(df_qsPriceLocal['Time'].to_numpy(), df_qsPriceLocal['State(2)'].to_numpy(), color='blue',
                 label='State 2')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # plt.tight_layout()
        # plt.savefig("pricesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        #             "Pricing.eps", format='eps', bbox_inches='tight')
        # plt.cla()
        plt.title('Local Pricing')
        print(df_qsPriceLocal)
        plt.show()

        # plt.figure(figsize=(7, 5), dpi=100)
        # plt.rc('axes', axisbelow=True)
        # plt.grid(lw=1.1)
        # plt.plot(df_qsRewAdjPriceLocal['Time'].to_numpy(), df_qsRewAdjPriceLocal['State(0)'].to_numpy(), color='green', label='State 0')
        # plt.plot(df_qsRewAdjPriceLocal['Time'].to_numpy(), df_qsRewAdjPriceLocal['State(1)'].to_numpy(), color='red', label='State 1')
        # plt.plot(df_qsRewAdjPriceLocal['Time'].to_numpy(), df_qsRewAdjPriceLocal['State(2)'].to_numpy(), color='blue', label='State 2')
        # plt.ylabel('Price-Adjusted Reward')
        # plt.xlabel('Time')
        # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # # plt.tight_layout()
        # # plt.savefig("pricesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        # #             "Pricing.eps", format='eps', bbox_inches='tight')
        # # plt.cla()
        # plt.title('Local Pricing')
        # plt.show()
    if TT <= 5e7 and (optimalOrLocal == 'optimal' or optimalOrLocal == 'both'):
        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)
        plt.plot(df_qsPriceOpt['Time'].to_numpy(), df_qsPriceOpt['State(0)'].to_numpy(), color='green', label='State 0')
        plt.plot(df_qsPriceOpt['Time'].to_numpy(), df_qsPriceOpt['State(1)'].to_numpy(), color='red', label='State 1')
        plt.plot(df_qsPriceOpt['Time'].to_numpy(), df_qsPriceOpt['State(2)'].to_numpy(), color='blue', label='State 2')
        plt.ylabel('Effective Price') # -Adjusted Reward
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # plt.tight_layout()
        # plt.savefig("pricesOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        #             "Pricing.eps", format='eps', bbox_inches='tight')
        # plt.cla()
        plt.title('Optimal Pricing')
        print(df_qsPriceOpt)
        print(df_qsRewAdjPriceOpt)
        plt.show()

        # plt.figure(figsize=(7, 5), dpi=100)
        # plt.rc('axes', axisbelow=True)
        # plt.grid(lw=1.1)
        # plt.plot(df_qsPrice['Time'].to_numpy(), df_qsPrice['State(0)'].to_numpy(), color='green', label='State 0')
        # plt.plot(df_qsPrice['Time'].to_numpy(), df_qsPrice['State(1)'].to_numpy(), color='red', label='State 1')
        # plt.ylabel('Price')
        # plt.xlabel('Time')
        # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # plt.tight_layout()
        # plt.savefig("pricesOverTimeLinearExclude2_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        #             "Pricing.eps", format='eps', bbox_inches='tight')
        # plt.cla()

    # df_qsRewAdjPrice.to_csv("pricesRewAdjOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
    #                         "Pricing.csv", index=False)
    # if TT <= 5e7:
    #     plt.figure(figsize=(7, 5), dpi=100)
    #     plt.rc('axes', axisbelow=True)
    #     plt.grid(lw=1.1)
    #     plt.plot(df_qsRewAdjPrice['Time'].to_numpy(), df_qsRewAdjPrice['State(0)'].to_numpy(),
    #              color='green', label='State 0')
    #     plt.plot(df_qsRewAdjPrice['Time'].to_numpy(), df_qsRewAdjPrice['State(1)'].to_numpy(),
    #              color='red', label='State 1')
    #     plt.plot(df_qsRewAdjPrice['Time'].to_numpy(), df_qsRewAdjPrice['State(2)'].to_numpy(),
    #              color='blue', label='State 2')
    #     plt.ylabel('Price-Adjusted Reward')
    #     plt.xlabel('Time')
    #     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    #     plt.tight_layout()
        # plt.savefig("pricesRewAdjOverTimeLinear_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        #             "Pricing.eps", format='eps', bbox_inches='tight')
        # plt.cla()
        # plt.show()

        # plt.figure(figsize=(7, 5), dpi=100)
        # plt.rc('axes', axisbelow=True)
        # plt.grid(lw=1.1)
        # plt.plot(df_qsRewAdjPrice['Time'].to_numpy(), df_qsRewAdjPrice['State(0)'].to_numpy(),
        #          color='green', label='State 0')
        # plt.plot(df_qsRewAdjPrice['Time'].to_numpy(), df_qsRewAdjPrice['State(1)'].to_numpy(),
        #          color='red', label='State 1')
        # plt.ylabel('Price-Adjusted Reward')
        # plt.xlabel('Time')
        # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # plt.tight_layout()
        # plt.savefig("pricesRewAdjOverTimeLinearExclude2_lambda" + str(lambdaVal) + "_" + optimalOrLocal +
        #             "Pricing.eps", format='eps', bbox_inches='tight')
        # plt.cla()

    print("took "+str(time() - start)[:6])


if __name__ == '__main__':

    plot_likeCDF()
    exit()

    # plotRatios_betaModel("mu")
    # exit()

    # simModulePriorsChange(5, 1)
    # exit()


    # Eren used plotRatios_likeCDF and simModuleLinear on 5.24.2023 to create the ML figures in the thesis
    # plotRatios_likeCDF()
    # exit()
    # simModuleLinear(5, 200)#, 0.2017818274585373 , 0.5293921710756335, 0.8660037919295616)
    # exit()

    # plot
    # plotRatios(9)  # argument is either 7 or 9, for \lambda = (1-\beta) / 0.7 or divided by 0.9, respectively
    # plotRatiosGrouped(9)
    # plotRatiosMu1Lambda(1)
    # exit()

    denemeSimulationLinear(TT=int(2e5), optimalOrLocal='both')  # 'optimal' or 'local' or 'both'
    # denemeSimulation()
    exit()

    # (a,b): if total transitions is x, then prior at least needs to be sth like (1, x-1)
    # (a,b,c): if total transitions is x, then prior at least needs to be sth like (1, 1, 3x-2),
    # e.g., 3 transition needs (1,1,7); 4 transition needs (1,1,10)
    # (a,b,c,d): if total transitions is x, then prior at least needs to be sth like (1, 1, 6x-3),
    # e.g., 3 transition needs (1,1,1,15); 4 transition needs (1,1,1,21)
    # (a,b,c,d,e): if total transitions is x, then prior at least needs to be sth like (1, 1, 9x-1),
    # e.g., 3 transition needs (1,1,1,1,26); 4 transition needs (1,1,1,1,36)

    # binary ratings OK: (beta 0.9, numTrans 65), (beta 0.8, numTrans 31), (beta 0.7, numTrans 20),
    # (beta 0.6, numTrans 14), (beta 0.5, numTrans 10), (beta 0.4, numTrans 8), (beta 0.3, numTrans 6)
    # (beta 0.2, numTrans 5), (beta 0.1, numTrans 3)
    # tertiary ratings: (beta 0.9, numTrans 44), (beta 0.8, numTrans 20) 5 each way (125 priors) took 3 hrs - all ok,
    # (beta 0.7, numTrans 13) k (took 1hr),
    # (beta 0.6, numTrans 9) k, (beta 0.5, numTrans 7) k, (beta 0.4, numTrans 6) k, (beta 0.3, numTrans 5) k
    # (beta 0.2, numTrans 4) k, (beta 0.1, numTrans 3) k

    beta_ = np.array([0.4])
    # priors go from (1,1) to whatever
    # for success-fail model, (a,b) holds for a fails and b successes
    numTrans_ = np.array([8])#, 10, 10])  # total no. of ratings that can be received, or total no. of transitions b4 reaching end
    rngStates__ = 5  # keep this as 5 for 5 star rating, if it's 2 then you have the beta-bernoulli model
    numPriors_ = 3 # should be very small for the general case, >2 state dimensions.
    for iij in range(len(beta_)):
        mainSim(beta_[iij], numTrans_[iij], rngStates__, numPriors_, size=1e5, fullModel=False)
    exit()


    # beta_ = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
    # mu_ = 1

    # mu = lambda = 1; 1-beta == 0.1 to 1, beta == 0 to 0.9, increment 0.1
    # 1 - beta on x-axis
    # 1-beta on y-axis, avg performance and minimum ratio over all instances
    # N = 25 or 30 for all beta
    # If beta = 0 there is nothing to do because LME = OPT :: x(0,0) = 1

    # 22 gives 10% error for beta 0.9
    # for success-fail model, (a,b) holds for a fails and b successes
    # numTrans_ = np.array([3, 5, 6, 8, 10, 14, 20, 31, 65])  # total no. of transitions before reaching the end
    # numTrans_ = np.ones(len(beta_)) * 30 # np.array([21]) #2, 3, 4, 6, 7, 10, 13, 21, 44])  # smaller epsilon, shorter horizons, larger error room
    # rngStates__ = 2  # keep this as 5 for 5 star rating, if it's 2 then you have the beta-bernoulli model
    # numPriors_ = 10  # should be very small for the general case, >2 state dimensions.

    # listOfRatios = np.zeros(int(numPriors_ * numPriors_ * len(beta_)))
    # counter = 0
    # start_time = time()
    # for st in range(len(beta_)):
    #     state = int(numTrans_[st] + 1)
    #     lambda_ = 0.5  # (1 - beta_[st]) / 0.9
    #     obj = np.zeros((state, state))
    #     transition = {}
    #
    #     for (m, n) in ((i, j) for (i, j) in product(range(numPriors_), range(numPriors_))):
    #         succ = m + 1
    #         fail = n + 1
    #         print("Prior is (" + str(succ) + ", " + str(fail) + "). No. transitions " + str(state - 1) + ".")
    #         for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
    #             obj[i][j] = (succ + i) / (succ + fail + i + j)
    #             transition[str(i) + "_" + str(j)] = {}
    #             if i + j < (state - 1):
    #                 transition[str(i) + "_" + str(j)]['s'] = obj[i][j]  # success
    #                 transition[str(i) + "_" + str(j)]['f'] = 1 - transition[str(i) + "_" + str(j)]['s']  # failure
    #                 transition[str(i) + "_" + str(j)]['r'] = 0  # remain
    #             else:
    #                 transition[str(i) + "_" + str(j)]['s'] = 0  # success
    #                 transition[str(i) + "_" + str(j)]['f'] = 0  # failure
    #                 transition[str(i) + "_" + str(j)]['r'] = 1
    #             print(i, j, "", transition[str(i) + "_" + str(j)].values())
                # if sum(transition[str(i) + "_" + str(j)].values()) != 1:
                #     exit("problem in (" + str(i) + ", " + str(j) + ")")
            #
            # slacks = np.ones((state, state)) * max(mu_, lambda_ / (1 - beta_[st]))
            # print("objective\n", obj)
            #
            # newSoln, _, objVal, mass = succfailOpt(n=len(obj), beta=beta_[st], lambd=lambda_, mu=mu_,
            #                                        prevSoln=np.empty(0), usePrevSoln=False,
            #                                        objective=obj, tr=transition, slacks=slacks, printResults=False)
            # print("Moving to fixed point thing.")
            # LME_soln, objVal_LME, _, _ = succfailFixedPoint(n=len(obj), beta=beta_[st], lambd=lambda_,
            #                                                 mu=min(mu_, mass), objective=obj, tr=transition,
            #                                                 bw="worst", printResults=False)
            # listOfRatios[counter] = objVal_LME / objVal
            # counter += 1
            # print("\nRatio is " + str(objVal_LME / objVal)[:7] + " at " + str(state - 1) + " instance size.", end=" ")
            # print("Prior was (" + str(succ) + ", " + str(fail) + ").")
            # print("Lambda: " + str(lambda_)[:5] + ", and ratio of mu/lambda: " + str(mu_/lambda_)[:5] + ".")
            # print("ratio of mu/lambda * (1-beta): " + str(mu_/lambda_ * (1 - beta_[st]))[:5] +
            #       ", beta: " + str(beta_[st])[:5] + ", (1-beta): " + str(1 - beta_[st])[:5] +
            #       ". Took " + str(time() - start_time)[:4] + " seconds.\n")
            # if counter % 5 == 0:
            #     print("Recorded results so far at " + str(datetime.now()))
            #     pd.DataFrame(listOfRatios).to_csv("LMEratios_mu1lambdaHalf_05betas.csv", index=False, header=False)
            # if objVal_LME / objVal < 1 - 1e-8 and state > 3:
            #     exit()

    # print("All ratios: ")
    # originalOptions = np.get_printoptions()
    # np.set_printoptions(threshold=np.inf)
    # print(listOfRatios)
    # np.set_printoptions(**originalOptions)
    # exit()
