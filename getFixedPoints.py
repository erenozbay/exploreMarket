from primals import *


def succfailFixedPoint(n, beta, lambd, mu, objective, tr, bw="best", printResults=True):
    # prints out the fixed point whenever it finds it
    # (0, 0) should be positive, hence the slacks for all states strictly better than (0, 0) should be zero
    slacks, bounds = np.zeros((n, n)), np.zeros((n, n))
    bounds[0][0] = mu
    
    minRew = 1
    for (i, j) in ((i, j) for j in range(n) for i in range(n) if (i + j <= n - 1)):
        if minRew >= objective[i][j]:
            minRew = objective[i][j]
        if objective[i][j] > objective[0][0]:
            slacks[i][j] = 1
            bounds[i][j] = mu

    if printResults:
        print("\ncalling the optimization model")
    
    # last argument 1 implies objective is maximize(a), last argument 0 implies objective is minimize(a)
    soln, feasibility, slacksSoln = succfailOpt(n, beta, lambd, mu, prevSoln=np.ones(1), usePrevSoln=False,
                                                objective=objective, tr=tr, slacks=slacks, printResults=printResults,
                                                bounds=bounds, whichobj=1)
    # succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objective, 1)
    soln_alt, feasibility_alt, slacksSoln = succfailOpt(n, beta, lambd, mu, prevSoln=np.ones(1), usePrevSoln=False,
                                                        objective=objective, tr=tr, slacks=slacks,
                                                        printResults=printResults, bounds=bounds, whichobj=0)
    # succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objective, 0)
    objectiveValue, objectiveValue_alt = 0, 0
    finalObjVal = 0 if bw == "best" else 1000
    changed_soln = 0
    both_solns = -1
    keepSoln = soln
    keepSlacks = slacksSoln
    if feasibility and feasibility_alt and printResults:
        print("Both initial problems are infeasible.\n")
    if not feasibility:
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectiveValue += objective[i][j] * soln[i][j]
        if printResults:
            print("maximize slack obj is feasible.")
            
    if not feasibility_alt:
        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
            objectiveValue_alt += objective[i][j] * soln_alt[i][j]
        if printResults:
            print("minimize reward obj is feasible.")
            
    if (objectiveValue > 0) & (objectiveValue_alt > 0):
        keepSoln = soln if objectiveValue_alt > objectiveValue else soln_alt
        if printResults:
            print("solution", keepSoln)
            print("slacks", slacksSoln)
        both_solns = 1
        finalObjVal = min(objectiveValue, objectiveValue_alt)
        if abs(objectiveValue - objectiveValue_alt) > 1e-6:
            changed_soln += 1
    elif objectiveValue > 0:
        both_solns = 0
        keepSoln = soln
        if printResults:
            print(keepSoln)
            print(slacksSoln)
        finalObjVal = objectiveValue
    elif objectiveValue_alt > 0:
        both_solns = 0
        keepSoln = soln_alt
        if printResults:
            print(keepSoln)
            print(slacksSoln)
        finalObjVal = objectiveValue_alt

    success_prob = objective[0][0]  # the success probability of (0, 0)
    counter = 0
    while (success_prob >= minRew) and (counter < n * n * 0.6):
        counter += 1
        nextSuccessProb = 0
        ind = -1
        possible_indices = np.zeros(((n + 1) // 2, 2))
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] <= success_prob)
                                                                      & (i + j <= n - 1))):
            if objective[i][j] >= success_prob:
                ind += 1
                possible_indices[ind][0] = i
                possible_indices[ind][1] = j
            else:
                if objective[i][j] >= nextSuccessProb:  # to get the next success_prob I can use
                    nextSuccessProb = objective[i][j]
        if printResults:
            print("nextSuccessProb", nextSuccessProb)
        if (possible_indices > 0).any():    # if you get no options above, directly move on to the next success_prob
                                            # enumerate all possible solutions with the slack choices

            if ind > 0:
                # for s in product([0, 1], repeat=(ind + 1)):  # potentially for various combination of slacks
                #     if sum(s) > 0:
                for inc in range(ind + 1):
                    ii = int(possible_indices[inc][0])
                    jj = int(possible_indices[inc][1])
                    # if s[inc] == 1:
                    bounds[ii][jj] = mu
                    # else:
                    #     bounds[ii][jj] = 0

                objectiveValue = 0
                # firstTime = 0
                for jk in range(2):
                    soln, feasibility, slacksSoln = succfailOpt(n, beta, lambd, mu, prevSoln=np.ones(1),
                                                                usePrevSoln=False, objective=objective, tr=tr,
                                                                slacks=slacks, printResults=printResults,
                                                                bounds=bounds, whichobj=jk)
                    # succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objective, jk)
                    if not feasibility:
                        condition = False
                        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                            objectiveValue += objective[i][j] * soln[i][j]
                            condition = objectiveValue >= finalObjVal if bw == "best" \
                                else objectiveValue <= finalObjVal
                        if np.sum(np.abs(keepSoln - soln)) > 1e-8 and condition:
                            if finalObjVal > 0:
                                changed_soln += 1
                                if printResults:
                                    print("changed solution.")
                            keepSoln = soln
                            keepSlacks = slacksSoln
                            finalObjVal = objectiveValue
                            if printResults:
                                print(keepSoln)
                                print(keepSlacks, "\n")
                    objectiveValue = 0
            else:  # just a single state slack update
                ii = int(possible_indices[ind][0])
                jj = int(possible_indices[ind][1])
                bounds[ii][jj] = mu

                objectiveValue = 0
                # firstTime = 0
                for jk in range(2):
                    soln, feasibility, slacksSoln = succfailOpt(n, beta, lambd, mu, prevSoln=np.ones(1),
                                                                usePrevSoln=False, objective=objective, tr=tr,
                                                                slacks=slacks, printResults=printResults,
                                                                bounds=bounds, whichobj=jk)
                            # succfailOptFixedPointPriors(n, beta, lambd, mu, bounds, slacks, objective, jk)
                    if not feasibility:
                        condition = False
                        for (i, j) in ((i, j) for j in range(n) for i in range(n)):
                            objectiveValue += objective[i][j] * soln[i][j]
                            condition = objectiveValue >= finalObjVal if bw == "best" \
                                else objectiveValue <= finalObjVal
                        if np.sum(np.abs(keepSoln - soln)) > 1e-8 and condition:
                            if finalObjVal > 0:
                                changed_soln += 1
                                if printResults:
                                    print("changed solution.")
                            keepSoln = soln
                            keepSlacks = slacksSoln
                            finalObjVal = objectiveValue
                            if printResults:
                                print(keepSoln)
                                print(keepSlacks, "\n")
                    objectiveValue = 0

        success_prob = nextSuccessProb
        # before moving on to the next success_prob, I need to fix the slacks and bounds for all those with higher probs
        for (i, j) in ((i, j) for j in range(n) for i in range(n) if ((objective[i][j] > success_prob)
                                                                      & (i + j <= n - 1))):
            slacks[i][j] = 1
            bounds[i][j] = mu

    if printResults:
        print("\nFinal",bw, "objval in fixed point is", finalObjVal, "; changed solns", changed_soln)
        print(keepSoln)
        print("slacks", keepSlacks)
    return keepSoln, finalObjVal, changed_soln, both_solns
