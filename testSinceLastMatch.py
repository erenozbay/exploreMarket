import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt


def simulationForLastMatch(state, T, workerarriveprob, jobarriveprob, wsp, bigK, rewardprob, C, _epsilon,
                           percent, recordEvery=10, pricing='lastMatch'):
    """
    This is the simulator for three pricing options:
        (1) Local queue length based
        (2) Optimal queue length based
        (3) Time since the queue was last emptied based

    See below (after the function) for details.
    Comments are left before/after corresponding lines

    Intended to be self-contained, only requires
        numpy, pandas, datetime, itertools and matplotlib.pyplot
    """

    " price calculation for the time since last emptied "

    def price_last_match(since_last_emptied, subtract_from, _epsilon, reward_cap=np.inf):
        return max(min(subtract_from - _epsilon * since_last_emptied, reward_cap), 0)

    " initialize all variables used for recording and tracking results "
    counter_conv, total_reward_tmp, counterr = 0, 0, 0
    queue, track_assign, queue_mid = np.zeros((state, state)), np.zeros((state, state)), np.zeros((state, state))
    (track_mass_tmp, track_queues_tmp) = (np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1)),
                                          np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1)))

    " 'since_last_match' keeps track of the time since some worker from a state has received a match "
    # since_last_match = np.zeros((state, state))

    track_price_since_last_match = np.zeros((int(T / recordEvery), int((state + 1) * state * 0.5) + 1))
    " number of columns is 1 more than needed because the first column is the 'effective' time "

    " vector of arrivals for workers "
    workerarrival = np.random.binomial(1, (np.ones(T) * workerarriveprob))
    print(f"Total worker arrivals will be {workerarrival.sum()}")

    " vector of arrivals for jobs "
    jobarrival = np.random.binomial(1, (np.ones(T) * jobarriveprob))
    print(f"Total job arrivals will be {jobarrival.sum()}")

    " some other parameters to keep track of assignments and pricing strategies "
    just_one, multiple, assignments_this_time = 0, 0, 0
    once_pricing = True

    " last time queue was empty "
    last_empty = np.zeros((state, state))
    since_last_empty, since_last_empty_tmp = np.zeros((state, state)), np.ones((state, state))
    last_match_price = np.zeros((state, state))
    eff_time = 0

    for t in range(T):
        queue[0][0] += workerarrival[t]  # arrivals to the initial state
        activeJobs = jobarrival[t]

        while (activeJobs >= 1) and (queue.sum() > 0):
            eff_time = t  # += 1  #
            activeJobs -= 1

            """
            maxval              keeps track of the highest price-adjusted-reward
            prev_maxval         keeps track of the 'most recent' highest price-adjusted-reward
            pos_i and pos_j     keep track of the state with the highest price-adjusted-reward
            eligible_states     is for randomizing the selection of states with identical price-adjusted-rewards

            we loop through each state one-by-one hence
                price
            represents the price associated with the current state
            """
            maxval = 0
            prev_maxval = -1
            pos_i, pos_j = state + 1, state + 1
            eligible_states = []
            price = np.inf

            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):

                " calculate the price of state (i, j) depending on the pricing strategy "
                if pricing == 'lastMatch':
                    price = price_last_match(since_last_emptied=since_last_empty[i][j], subtract_from=C,
                                             _epsilon=_epsilon)
                    if since_last_empty_tmp[i][j] == 0:
                        price = last_match_price[i][j]

                    if once_pricing:
                        print("Pricing is time since last match based.")
                        once_pricing = False

                elif pricing == 'local':
                    price = C * (bigK - min(bigK, queue[i][j])) / bigK

                    if once_pricing:
                        print("Pricing is queue length based.")
                        once_pricing = False

                elif pricing == 'optimal':
                    if i + j < (state - 1):
                        price = (bigK - min(bigK, queue[i][j])) / bigK - \
                                wsp * (i + 1) / (i + j + 2) * (bigK - min(bigK, queue[i + 1][j])) / bigK - \
                                wsp * (j + 1) / (i + j + 2) * (bigK - min(bigK, queue[i][j + 1])) / bigK
                    elif i + j == (state - 1):
                        price = (bigK - min(bigK, queue[i][j])) / bigK - \
                                wsp * (bigK - min(bigK, queue[i][j])) / bigK

                    price = C * price

                    if once_pricing:
                        print("Pricing is optimal.")
                        once_pricing = False

                " Pick a non-empty state with a non-negative price-adjusted-reward "
                if (maxval <= (rewardprob[i][j] - price)) & (queue[i][j] > 0):
                    """
                    Store the price-adjusted reward in maxval and compare it with the prev_maxval
                    Scenarios:
                        (1) For the first time of entry in this if-statement, maxval > prev_maxval
                            Hence, eligible_states = [[pos_i, pos_j]]
                            (a) Same applies for overwriting multiple identical price-adjusted-reward states with a
                                state strictly better than them, i.e., initialize the eligible states with the new
                                better state
                        (2) For multiple entries with the same price-adjusted-reward value,
                            eligible_states.append([pos_i, pos_j])
                            is executed
                            (a) If a new strictly better state pops up, Scenario 1a is returned.
                    """

                    maxval = rewardprob[i][j] - price

                    pos_i = i
                    pos_j = j

                    if maxval == prev_maxval:
                        eligible_states.append([pos_i, pos_j])
                    elif maxval > prev_maxval:
                        eligible_states = [[pos_i, pos_j]]

                    prev_maxval = maxval

            " pick from multiple states if applicable, count occurrences for reporting "
            if len(eligible_states) > 1:
                picked_index = np.random.randint(0, len(eligible_states) - 1)
                pos_i = eligible_states[picked_index][0]
                pos_j = eligible_states[picked_index][1]
                multiple += 1
            elif len(eligible_states) == 1:
                pos_i = eligible_states[0][0]
                pos_j = eligible_states[0][1]
                just_one += 1

            " if there is a state that is picked to receive a match, proceed "
            if len(eligible_states) > 0:

                queue[pos_i][pos_j] -= 1
                track_assign[pos_i][pos_j] += 1

                if (queue < 0).any():
                    print("Oops, a non-existent worker left. Exiting...")
                    break

                # epsilon_up = False
                " Last match price for the selected state "
                if (since_last_empty_tmp[pos_i][pos_j] == 0) or (queue[pos_i][pos_j] == 0):
                    last_match_price[pos_i][pos_j] = last_match_price[pos_i][pos_j] + _epsilon * 10
                # elif queue[pos_i][pos_j] == 0:
                #     last_match_price[pos_i][pos_j] = last_match_price[pos_i][pos_j] + _epsilon * 10
                else:
                    last_match_price[pos_i][pos_j] = price_last_match(
                        since_last_emptied=since_last_empty[pos_i][pos_j],
                        subtract_from=C,
                        _epsilon=_epsilon
                    )

                if queue[pos_i][pos_j] == 0:
                    last_empty[pos_i][pos_j] = eff_time
                    " Price when a state hits zero queue length shouldn't skyrocket "
                    " Let it be (2 * price) or (price + epsilon) "
                    since_last_empty_tmp[pos_i][pos_j] = 0
                    # if not epsilon_up:
                    #     last_match_price[pos_i][pos_j] = last_match_price[pos_i][pos_j] + _epsilon * 10
                # else:
                #     " if there is match, price may remain the same "
                #     last_empty[pos_i][pos_j] += 1

                " sample the reward "
                reward = np.random.binomial(1, np.random.beta(pos_i + 1, pos_j + 1))

                " record rewards during the last (100-percent) percent of the time horizon "
                if t > T * percent / 100:
                    total_reward_tmp += reward

                " sample a binary to see if the selected worker will remain in the system or not "
                stay = np.random.binomial(1, wsp)

                " if the worker is staying, then depending on the reward and the current state, they move or remain "
                if stay == 1:
                    if pos_i + pos_j < (state - 1):
                        queue[pos_i + 1][pos_j] += reward
                        queue[pos_i][pos_j + 1] += (1 - reward)
                    else:
                        queue[pos_i][pos_j] += 1

                " count the total assignments throughout the time horizon for tracking purposes "
                assignments_this_time += 1

            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                if queue[i][j] == 0:
                    last_empty[i][j] += 1  # if the queue is at zero and have been for some time
                                                 # its price shouldn't decrease
                since_last_empty[i][j] = eff_time - last_empty[i][j]
                if queue[i][j] > 0:
                    since_last_empty_tmp[i][j] = eff_time - last_empty[i][j]


        " at most bigK many workers allowed in each queue at any time "
        for (i, j) in product(range(state), range(state)):
            queue[i][j] = min(queue[i][j], bigK)

        " keep track of the results every 'recordEvery' time period"
        if int((t + 1) / recordEvery) == ((t + 1) / recordEvery):
            track_mass_tmp[counter_conv][0] = counter_conv + 1
            track_queues_tmp[counter_conv][0] = counter_conv + 1
            track_price_since_last_match[counter_conv][0] = counter_conv + 1

            index = 0
            for (i, j) in ((i, j) for (i, j) in product(range(state), range(state)) if i + j <= (state - 1)):
                track_mass_tmp[counter_conv][index + 1] = track_assign[i][j] / (t + 1)
                track_queues_tmp[counter_conv][index + 1] = queue[i][j]

                track_price_since_last_match[counter_conv][index + 1] = last_match_price[i][j]

                # track_price_since_last_match[counter_conv][index + 1] = (
                #     price_last_match(since_last_emptied=since_last_empty[i][j], subtract_from=C,
                #                      _epsilon=epsilon))
                index += 1

            counter_conv += 1

    " normalize the reward "
    total_reward_tmp = total_reward_tmp / (T * (1 - percent / 100))

    print(f"Total assignments {assignments_this_time}, with a time horizon of {T} out of {jobarrival.sum()} jobs.")
    print(f"just_one {just_one}, multiple {multiple}, total reward {total_reward_tmp}.")
    return track_mass_tmp, total_reward_tmp, track_queues_tmp, track_price_since_last_match


"""
This is the following tree model

State 0 --> State 1 --> State 2
  |            |
  v            v
State 3 --> State 4
  |
  v
State 5

State 0 --> State (0,0) --> Reward: 0.5 --> LME price: 0   --> LME match rate: 0.0476
State 1 --> State (0,1) --> Reward: 0.4 --> LME price: 0   --> LME match rate: 0
State 2 --> State (0,2) --> Reward: 0.3 --> LME price: >=0 --> LME match rate: 0
State 3 --> State (1,0) --> Reward: 0.7 --> LME price: 0.2 --> LME match rate: 0.0224
State 4 --> State (1,1) --> Reward: 0.6 --> LME price: 0.1 --> LME match rate: 0.143
State 5 --> State (2,0) --> Reward: 0.8 --> LME price: 0.3 --> LME match rate: 0.287

State 0 --> Reward: 0.5 --> OPT effective price: -0.254 --> Optimal match rate: 0.067
State 1 --> Reward: 0.4 --> OPT effective price: 0      --> Optimal match rate: 0
State 2 --> Reward: 0.3 --> OPT effective price: 0      --> Optimal match rate: 0
State 3 --> Reward: 0.7 --> OPT effective price: -0.054 --> Optimal match rate: 0.032
State 4 --> Reward: 0.6 --> OPT effective price: 0      --> Optimal match rate: 0
State 5 --> Reward: 0.8 --> OPT effective price: 0.046  --> Optimal match rate: 0.401

OPT Reward: 0.3767
LME Reward: 0.35488

lambda = 0.1
mu = 0.5
beta = 0.95
"""

numState = 3
" reward values "
objVals = np.zeros((numState, numState))
objVals[0][0], objVals[0][1], objVals[0][2] = 0.5, 0.4, 0.3
objVals[1][0], objVals[1][1] = 0.7, 0.6
objVals[2][0] = 0.8

" lambda "
workerArriveProbability = 0.1

" beta "
workerStayProbability = 0.95

" mu "
jobArriveProbability = 0.5

"""
pricing_strategy can be 'lastMatch' or 'local' or 'optimal'

if 'lastMatch' --> price(i, j) = price_multiplier - epsilon * time_since_last_matched(i, j)
                   where time_since_last_matched(i, j) is incremented by one each time period state (i, j) does not
                   receive a match; time_since_last_matched(i, j) is initialized to zero once state (i, j) receives a
                   match.
                   objVals[numState - 1][0] is the highest reward --> 0.8 in this case
                   
                   IF (time since last matched initialized to zero) THEN MATCH RATES AND PRICES SUCK
                   IF (time since last matched reduced by one) THEN ONLY PRICES SUCK, MATCH RATES match THE LME

if 'local'     --> price(i, j) = ( 1 - min(1, queue_length(i, j) / K) ) 
                                 * price_multiplier

if 'optimal'   --> if in a middle state
                        price(i, j) = ( 1 - min(1, queue_length(i, j) / K)
                                        - beta * (i + 1) / (i + j + 2) * (1 - min(1, queue_length(i + 1, j) / K))
                                        - beta * (j + 1) / (i + j + 2) * (1 - min(1, queue_length(i, j + 1) / K)) )
                                       * price_multiplier

                    if in an edge state
                        price(i, j) = ( 1 - min(1, queue_length(i, j) / K)
                                        - beta * (1 - min(1, queue_length(i, j) / K)) ) 
                                      * price_multiplier

'epsilon', 'price_multiplier', and 'K' are defined below.

For 'local' and 'optimal' use 
        price_multiplier = 2 * objVals[numState - 1][0]
        K = 1000
        
'epsilon' is used only when 
        pricing_strategy = 'lastMatch'  
"""
pricing_strategy = 'lastMatch'
epsilon = 0.001
price_multiplier = 1 * objVals[numState - 1][0]  # objVals[numState - 1][0] - objVals[0][numState - 1]  # 1
K = 1000

""" 
The time horizon
        A time horizon of 5e6 (5*10^6) takes 130 seconds for local
                                             150 seconds for optimal 
                                             160 seconds for lastMatch
Use
    time_horizon = int(5e6)
"""
time_horizon = int(5e6)
"""
The results are not recorded for each time period but recorded every 'recordEveryPeriod' periods
This makes the 'effective' time of the simulation as (time_horizon / recordEveryPeriod)
Use
    recordEveryPeriod = 10
"""
recordEveryPeriod = 10

"""
This means that the rewards will be recorded only after this much percent of the time horizon has passed
Use 
    recordRewardsAfterThisPercent = 80
"""
recordRewardsAfterThisPercent = 80

start = datetime.now()  # note the starting time

(track_mass, total_reward,
 track_queues, track_last_match_prices) = simulationForLastMatch(numState, time_horizon, workerArriveProbability,
                                                                 jobArriveProbability, workerStayProbability, K,
                                                                 objVals, price_multiplier, epsilon,
                                                                 recordRewardsAfterThisPercent,
                                                                 recordEveryPeriod, pricing_strategy)

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

df_qsLastMatch = pd.DataFrame(track_last_match_prices,
                              columns=['Time'.split() + ['Price(' + str(i) + ',' + str(j) + ')' for (i, j) in
                                                         ((i, j) for (i, j) in
                                                          product(range(numState), range(numState)) if
                                                          i + j <= (numState - 1))]], dtype=float)

" Calculate local queue length based prices using queue lengths during simulation "
for (i, j) in ((i, j) for (i, j) in product(range(numState), range(numState)) if i + j <= (numState - 1)):
    name = 'State(' + str(i) + ',' + str(j) + ')'
    namep = 'Price(' + str(i) + ',' + str(j) + ')'

    # local price #
    df_qsTree[namep] = df_qsTree.apply(lambda x: price_multiplier * (K - min(K, x[name])) / K, axis=1)
    # local price #

" Print stuff for the sake of printing "
print("Match rates")
print(df_massTree)

print("\n\n")

print("Local queue length based prices")
print(df_qsTree)

print(f"Took {(datetime.now() - start).total_seconds()} seconds.")

print("\n\n")

print("Time since last matched based prices")
print(df_qsLastMatch)

"""
Create and save figures for the
    Match rates
    Queue length based prices
    Time since last matched based prices
"""

plt.figure(figsize=(7, 5), dpi=100)
plt.rc('axes', axisbelow=True)
plt.grid(lw=1.1)

plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(0,0)'].to_numpy(), color='pink', label='Match Rate of 0')
plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(0,1)'].to_numpy(), color='orange', label='Match Rate of 1')
plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(0,2)'].to_numpy(), color='green', label='Match Rate of 2')
plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1,0)'].to_numpy(), color='cyan', label='Match Rate of 3')
plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(1,1)'].to_numpy(), color='blue', label='Match Rate of 4')
plt.plot(df_massTree['Time'].to_numpy(), df_massTree['State(2,0)'].to_numpy(), color='purple', label='Match Rate of 5')

plt.ylabel('Match Rates')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig("Masses.eps", format='eps', bbox_inches='tight')
plt.close()
# plt.show()

"""
Queue length based
"""
plt.figure(figsize=(7, 5), dpi=100)
plt.rc('axes', axisbelow=True)
plt.grid(lw=1.1)

plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(0,0)'].to_numpy(), color='pink', label='Price of 0')
plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(0,1)'].to_numpy(), color='orange', label='Price of 1')
plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(0,2)'].to_numpy(), color='green', label='Price of 2')
plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(1,0)'].to_numpy(), color='cyan', label='Price of 3')
plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(1,1)'].to_numpy(), color='blue', label='Price of 4')
plt.plot(df_qsTree['Time'].to_numpy(), df_qsTree['Price(2,0)'].to_numpy(), color='purple', label='Price of 5')

plt.ylabel('Queue Length Based Price')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig("QueueBasedPrices.eps", format='eps', bbox_inches='tight')
plt.close()
# plt.show()

"""
Time since last matched
"""

plt.figure(figsize=(7, 5), dpi=100)
plt.rc('axes', axisbelow=True)
plt.grid(lw=1.1)

plt.plot(df_qsLastMatch['Time'].to_numpy(), df_qsLastMatch['Price(0,0)'].to_numpy(), color='pink', label='Price of 0')
plt.plot(df_qsLastMatch['Time'].to_numpy(), df_qsLastMatch['Price(0,1)'].to_numpy(), color='orange', label='Price of 1')
plt.plot(df_qsLastMatch['Time'].to_numpy(), df_qsLastMatch['Price(0,2)'].to_numpy(), color='green', label='Price of 2')
plt.plot(df_qsLastMatch['Time'].to_numpy(), df_qsLastMatch['Price(1,0)'].to_numpy(), color='cyan', label='Price of 3')
plt.plot(df_qsLastMatch['Time'].to_numpy(), df_qsLastMatch['Price(1,1)'].to_numpy(), color='blue', label='Price of 4')
plt.plot(df_qsLastMatch['Time'].to_numpy(), df_qsLastMatch['Price(2,0)'].to_numpy(), color='purple', label='Price of 5')

plt.ylabel('Price')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig("LastMatchBasedPrices.eps", format='eps', bbox_inches='tight')
plt.close()
# plt.show()

if pricing_strategy == 'lastMatch':

    pricess = ['Price(0,0)', 'Price(0,1)', 'Price(0,2)', 'Price(1,0)', 'Price(1,1)', 'Price(2,0)']
    labelss = ['Price of 0', 'Price of 1', 'Price of 2', 'Price of 3', 'Price of 4', 'Price of 5']

    for ind in range(len(pricess)):
        plt.figure(figsize=(7, 5), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid(lw=1.1)

        plt.plot(df_qsLastMatch['Time'].to_numpy(), df_qsLastMatch[pricess[ind]].to_numpy(), label=labelss[ind])

        # plt.plot(df_qsLastMatch['Time'].to_numpy(),
        #          np.cumsum(
        #              df_qsLastMatch[pricess[ind]].to_numpy()
        #          ) / np.arange(1, len(df_qsLastMatch[pricess[ind]].to_numpy()) + 1), label=labelss[ind])

        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("LastEmptiedBasedPrices_" + labelss[ind] + ".eps", format='eps', bbox_inches='tight')
        plt.show()
        plt.close()
