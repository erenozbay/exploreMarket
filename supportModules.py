import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick

def plotRatiosMu1Lambda(oneOrHalf):
    if oneOrHalf == 1:
        ratios = pd.read_csv('LMEratios_mu1lambda1.csv').to_numpy()
        oneOrHalf = 'One'
    elif oneOrHalf == 1/2:
        ratiosPoint1 = pd.read_csv('LMEratios_mu1lambdaHalf_betas5_6_7_8_9.csv').to_numpy()
        ratiosPoint05 = pd.read_csv('LMEratios_mu1lambdaHalf_05betas.csv').to_numpy()
        ratios = np.zeros((1000, 1))
        for i in range(10):
            if i % 2 == 0:
                j = int(i / 2)  # 0 to 0; 2 to 1; 4 to 2; 6 to 3; 8 to 4
                ratios[(i * 100):((i + 1) * 100)] = ratiosPoint1[(j * 100):((j + 1) * 100)]
            else:
                j = int((i - 1) / 2)  # 1 to 0; 3 to 1; 5 to 2; 7 to 3; 9 to 4
                ratios[(i * 100):((i + 1) * 100)] = ratiosPoint05[(j * 100):((j + 1) * 100)]
        oneOrHalf = 'Half'
    else:
        exit("Weird input.")
    grouped = {}
    minRatios = np.zeros(10)
    avgRatios = np.zeros(10)
    oneMinusBeta = np.arange(1, 11) / 10

    for i in range(10):
        grouped['beta_' + str(i)] = ratios[(i * 100):((i + 1) * 100)]
        minRatios[i] = min(grouped['beta_' + str(i)])
        avgRatios[i] = np.mean(grouped['beta_' + str(i)])
    minRatiosFlipped = np.flip(minRatios)
    avgRatiosFlipped = np.flip(avgRatios)

    # print(grouped)
    # print(len(grouped))
    # for i in range(10):
    #     print(len(grouped['beta_' + str(i)]))
    # print(avgRatiosFlipped)
    # print(minRatiosFlipped)
    # print(oneMinusBeta)
    # exit()

    plt.figure(figsize=(7, 5), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid(lw=1.1)  # (axis="y")

    plt.plot(oneMinusBeta, minRatiosFlipped, color='r', marker='o', markersize=5.5, label="Minimum ratio", linewidth=2.6)
    plt.plot(oneMinusBeta, avgRatiosFlipped, color='b', marker='o', markersize=5.5, label="Average ratio", linewidth=2.6)
    plt.xticks(oneMinusBeta, fontsize=14)
    plt.yticks(fontsize = 13)

    # plt.plot(oneMinusBeta, oneMinusBeta, color='r')
    # plt.yticks(oneMinusBeta)
    # plt.ylim(oneMinusBeta[-3])
    plt.ylabel('Ratio', fontsize=20)
    plt.xlabel(r"$\frac{\mathrm{Demand}}{\mathrm{Supply}}$", fontsize=24)
    # plt.title('Ratios of LME Reward Across All Priors to OPT Reward', fontsize=13)
    title = 'newRatios_noLine_minAvg_lambda' + str(oneOrHalf) + '.eps'  # newRatios_noLine
    plt.legend(loc="lower right", prop={'size': 15})

    plt.savefig(title, format='eps', bbox_inches='tight')
    # plt.show()
    plt.cla()

def plotRatios_likeCDF():  # for CDF-like plots, use vals and edges
    # divideBy = betaVal
    filename = 'newFeedbackFigs/feedbackLMEvsOPTobjvalsAndPriceRatiosMLB_200sims_5states_priceDevs.csv'
    priceDev_B = pd.read_csv(filename).to_numpy()
    filename = 'newFeedbackFigs/feedbackLMEvsOPTobjvalsAndPriceRatiosMLB_200sims_5states_rewards.csv'
    rews_B = pd.read_csv(filename).to_numpy()
    filename = 'newFeedbackFigs/feedbackLMEvsOPTobjvals_PriceRatiosMLA_200sims_last10Percent_priceDevs.csv'
    priceDev_A = pd.read_csv(filename).to_numpy()
    filename = 'newFeedbackFigs/feedbackLMEvsOPTobjvals_PriceRatiosMLA_200sims_last10Percent_rewards.csv'
    rews_A = pd.read_csv(filename).to_numpy()

    use = rews_B
    # print(priceDev_B)
    title = 'rews_B.eps'
    xlab = 'Ratio of simulated and optimal rewards' #'Price Deviation (%)' #
    ylab = 'Ratio'
    # generate bins and bin edges
    vals, edges = np.histogram(use, bins=200)
    # cumsum and normalize to get cdf rather than pdf
    vals = np.cumsum(vals)
    vals = vals / vals[-1]
    # convert bin edges to bin centers
    edges = (edges[:-1] + edges[1:]) / 2
    print(edges)
    print(vals)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.grid(axis="y")
    lineWidth = 2.5
    plt.plot(edges, vals, color='b', linewidth=lineWidth)
    plt.axvline(x=np.mean(use), linestyle='dashed', color='r', linewidth=lineWidth * 0.8)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.xticks(fontsize=15)  #np.array([.97, 1, 1.03]),
    plt.yticks(fontsize=15)

    # xt = plt.gca().get_xticks()
    # xt = np.append(xt, np.mean(priceDev_B))
    #
    # xtl = xt.tolist()
    # xtl[-1] = str(np.mean(priceDev_B))[2] + "%"
    # plt.gca().set_xticks(xt)
    # plt.gca().set_xticklabels(xtl)

    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    # if divideBy == 9:
    #     plt.xticks(np.arange(0.9995, 1.0001, 0.0001))
    #     # plt.text(np.mean(ratios), 0, str(np.mean(ratios))[:7], rotation=90)
    #     # plt.xlim(0.999, 1.0001)
    # elif divideBy == 7:
    #     plt.xticks(np.arange(0.9975, 1.0002, 0.0005))
        # plt.text(np.mean(ratios), 0, str(np.mean(ratios))[:7], rotation=90)
        # plt.xlim(0.9972, 1.0001)
    # plt.ylabel('', fontsize=14)
    # plt.title('Ratio of Worst LME Reward to OPT Reward', fontsize=13)



    # plt.savefig(title, format='eps', bbox_inches='tight')
    plt.show()
    plt.cla()


def plot_likeCDF():  # for CDF-like plots, use vals and edges
    # divideBy = betaVal
    filename = 'taskrabbit/Taskrabbit-dataset_raw.csv'
    data = pd.read_csv(filename)
    # AverageRating   Number_of_reviews   Number_of_jobs   Price   Sex
    avgRating = data['AverageRating'].to_numpy()
    # print(avgRating)
    NOR = data['Number_of_reviews'].to_numpy()
    logNOR = np.log10(NOR + 1)
    # print(logNOR)
    NOJ = data['Number_of_jobs'].to_numpy()
    # print(NOJ)
    price = data['Price'].to_numpy()
    print(price)
    unique, counts = np.unique(price, return_counts=True)
    print(unique)
    print(counts)
    print(np.cumsum(counts))




    title = 'taskrabbit_prices.eps'
    xlab = 'Price' #'Price Deviation (%)' #
    ylab = 'Cumulative Count'
    # generate bins and bin edges
    vals, edges = np.histogram(price, bins=20)
    # cumsum and normalize to get cdf rather than pdf
    vals = np.cumsum(vals)
    vals = vals / vals[-1]
    # convert bin edges to bin centers
    edges = (edges[:-1] + edges[1:]) / 2
    print(edges)
    print(vals)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.grid(lw=0.25)
    lineWidth = 1
    # plt.plot(edges, vals, color='b', linewidth=lineWidth, marker='o', markersize=6)
    # plt.axvline(x=np.mean(price), linestyle='dashed', color='r', linewidth=lineWidth * 0.8)
    # plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # plt.scatter(x=logNOR, y=price, color='blue', marker='o', s=64) #, label=r"$\log$(number of reviews)")
    # plt.axline(xy1=(0, 35.59), slope=12.43, color ='red', linestyle='--', linewidth=2.5)
    # plt.legend(loc="upper right")
    # plt.xticks(fontsize=14)  # np.array([.97, 1, 1.03]),
    # plt.yticks(fontsize=14)
    # plt.xlabel(r"$\log$(number of reviews)", fontsize=16)
    # plt.ylabel('Price', fontsize=16)
    # plt.savefig("log(NOR)_vs_priceRegression.eps", format='eps', bbox_inches='tight')
    # plt.show()
    # plt.cla()
    # exit()

    # countsExcluded = counts[counts > 1]
    # unique = unique[counts > 1]
    # counts = countsExcluded
    # cumulativeSum = np.cumsum(counts)
    # plt.plot(unique.astype('str'), cumulativeSum, color='b', marker='o', markersize=4)
    # plt.xticks(fontsize=3)  #np.array([.97, 1, 1.03]),
    # plt.yticks(fontsize=7)
    # cumulativeSum = cumulativeSum[counts > 1]
    # set_yticks = cumulativeSum
    # plt.yticks(np.append(set_yticks, 120))
    # plt.xlabel(xlab, fontsize=16)
    # plt.ylabel(ylab, fontsize=16)


    # plt.savefig(title, format='eps', bbox_inches='tight')
    # plt.show()
    # plt.cla()


    # plt.bar(unique.astype('str'), counts, color='b')
    # plt.show()
    # plt.cla()

    freq = counts / sum(counts)
    print(freq)
    cumFreq = np.cumsum(freq)
    print(cumFreq)
    # plt.step(y=cumFreq, x=unique)
    # unique = np.log10(unique)
    plt.plot(np.append(unique[0], unique), np.append(0, cumFreq), drawstyle='steps', linewidth=2)
    plt.xticks(fontsize=14)  # np.array([.97, 1, 1.03]),
    plt.yticks(fontsize=14)
    plt.xlabel("Price", fontsize=16)  # r"$\log$(Price)"
    plt.savefig("cumulativePrices_step_cdf.eps", format='eps', bbox_inches='tight')
    plt.show()
    exit()

    plt.xticks(fontsize=14)  # np.array([.97, 1, 1.03]),
    plt.yticks(fontsize=14)
    plt.xlabel('Price', fontsize=16) # r"$\log$(Price)"
    plt.ylabel('Count', fontsize=16)
    rngStart = 1
    rngEnd = counts[0] + 1
    # unique = np.log10(unique)
    # plt.gca().set_xscale('log')
    for i in range(44):
        if i % 2 == 0:
            plt.scatter(x=np.repeat(unique[i], counts[i]) , y=np.arange(rngStart, rngEnd), color='blue', marker='.',
                        s=32)
        else:
            plt.scatter(x=np.repeat(unique[i], counts[i]), y=np.arange(rngStart, rngEnd), color='red', marker='v',
                        s=32)
        rngStart = rngEnd
        rngEnd = rngEnd + counts[i + 1] if i < 43 else 121



    plt.savefig("cumulativePrices.eps", format='eps', bbox_inches='tight')
    plt.show()
    plt.cla()


def plotRatios(betaVal):
    divideBy = betaVal
    ratios = pd.read_csv('LMEratios_divideByPoint' + str(divideBy) + '_shorterHorizons_missingPoint9Beta.csv').to_numpy()

    # generate bins and bin edges
    vals, edges = np.histogram(ratios, bins=800)
    # cumsum and normalize to get cdf rather than pdf
    vals = np.cumsum(vals)
    vals = vals / vals[-1]
    # convert bin edges to bin centers
    edges = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(8, 6), dpi=100)
    plt.grid(axis="y")
    plt.plot(edges, vals, color='b')
    plt.axvline(x=np.mean(ratios), ymin=0.01, ymax=0.956, linestyle='dashed', color='r')

    plt.xlabel('Ratio', fontsize=12)
    if divideBy == 9:
        plt.xticks(np.arange(0.9995, 1.0001, 0.0001))
        # plt.text(np.mean(ratios), 0, str(np.mean(ratios))[:7], rotation=90)
        # plt.xlim(0.999, 1.0001)
    elif divideBy == 7:
        plt.xticks(np.arange(0.9975, 1.0002, 0.0005))
        # plt.text(np.mean(ratios), 0, str(np.mean(ratios))[:7], rotation=90)
        # plt.xlim(0.9972, 1.0001)
    # plt.ylabel('', fontsize=14)
    # plt.title('Ratio of Worst LME Reward to OPT Reward', fontsize=13)
    title = 'ratiosFor' + str(divideBy) + '.eps'


    plt.savefig(title, format='eps', bbox_inches='tight')
    plt.show()
    plt.cla()


def plotRatiosGrouped(betaVal):
    divideBy = betaVal
    ratios = pd.read_csv('LMEratios_divideByPoint' + str(divideBy) + '_shorterHorizons_missingPoint9Beta.csv').to_numpy()
    grouped = {}
    labels = {}
    for i in range(8):
        grouped['beta_0' + str(i + 1)] = ratios[(i * 100):((i + 1) * 100)]
        labels[str(i)] = r"$\beta$ = 0." + str(i + 1)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid(axis="y")


    for i in range(8):
        plt.scatter(np.arange(i * 100 + 1 + 20 * i * (i > 0), (i + 1) * 100 + 1 + 20 * i * (i > 0)),
                              grouped['beta_0' + str(i + 1)], s=12, label=labels[str(i)])
    plt.ylabel('Ratio', fontsize=14)
    plt.xlabel('Prior', fontsize=14)

    plt.gca().axes.get_xaxis().set_ticks([])
    if divideBy == 9:
        plt.yticks(np.arange(0.9995, 1.0001, 0.0001))
        plt.ylim(bottom=0.9994)
    elif divideBy == 7:
        plt.yticks(np.arange(0.9975, 1.0001, 0.0005))
        # plt.ylim(bottom=0.9994)
    plt.title('Ratio of Worst LME Reward to OPT Reward', fontsize=13)
    title = 'ratiosGroupedFor' + str(divideBy) + '.eps'
    plt.legend(loc="lower left", prop={'size': 10}, ncol=2)#, bbox_to_anchor=(1, 1))

    plt.savefig(title, format='eps', bbox_inches='tight')
    plt.show()
    plt.cla()

def plotRatios_betaModel(across):
    if across == "priors":
        filename = 'forFigs3point7_inThesis/binaryBetaModel_5priors.csv'
        ratios = pd.read_csv(filename).to_numpy()
        LME_OPT = np.flip(ratios[:, 0])
        sim_OPT = np.flip(ratios[:, 1])
        print(ratios)
        print(LME_OPT)
        print(sim_OPT)
        # exit()
        # grouped = {}
        # labels = {}
        # for i in range(8):
        #     grouped['beta_0' + str(i + 1)] = ratios[(i * 100):((i + 1) * 100)]
        #     labels[str(i)] = r"$\beta$ = 0." + str(i + 1)

        plt.figure(figsize=(8, 6), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid()

        muVals = np.arange(1, 10) + 0.1

        plt.plot(muVals, LME_OPT, label='LME / OPT',
                 color='r', marker='o', markersize=5.5, linewidth=2.6)
        plt.plot(muVals, sim_OPT, label='Sim. / OPT',
                 color='blue', marker='o', markersize=5.5, linewidth=2.6)

        # for i in range(8):
        #     plt.scatter(np.arange(i * 100 + 1 + 20 * i * (i > 0), (i + 1) * 100 + 1 + 20 * i * (i > 0)),
        #                 grouped['beta_0' + str(i + 1)], s=12, label=labels[str(i)])
        plt.xlabel(r"$\mu$", fontsize=20)
        plt.ylabel('Ratio', fontsize=20)
        plt.xticks(muVals, fontsize=16)  # np.array([.97, 1, 1.03]),
        plt.yticks(fontsize=16)
        plt.ylim(bottom=0.93, top=1.005)
        # plt.gca().axes.get_xaxis().set_ticks([])
        # if divideBy == 9:
        #     plt.yticks(np.arange(0.9995, 1.0001, 0.0001))
        #     plt.ylim(bottom=0.9994)
        # elif divideBy == 7:
        #     plt.yticks(np.arange(0.9975, 1.0001, 0.0005))
            # plt.ylim(bottom=0.9994)
        # plt.title('Ratio of Worst LME Reward to OPT Reward', fontsize=13)
        # title = 'ratiosGroupedFor' + str(divideBy) + '.eps'
        plt.legend(loc="lower right", prop={'size': 16})  # , bbox_to_anchor=(1, 1))

        plt.savefig('acrossPriors.eps', format='eps', bbox_inches='tight')
        plt.show()
        plt.cla()
    elif across == "mu":
        filename = 'forFigs3point7_inThesis/binaryBetaModel_perPrior.csv'
        ratios = pd.read_csv(filename).to_numpy()
        LME_OPT = ratios[:, 0]
        sim_OPT = ratios[:, 1]
        print(ratios)
        print(LME_OPT)
        print(sim_OPT)
        # exit()
        # grouped = {}
        # labels = {}
        # for i in range(8):
        #     grouped['beta_0' + str(i + 1)] = ratios[(i * 100):((i + 1) * 100)]
        #     labels[str(i)] = r"$\beta$ = 0." + str(i + 1)

        plt.figure(figsize=(8, 6), dpi=100)
        plt.rc('axes', axisbelow=True)
        plt.grid()

        muVals = ['(1,1)', '(1,2)', '(1,3)', '(1,4)', '(1,5)']

        plt.plot(muVals, LME_OPT, label='LME / OPT',
                 color='r', marker='o', markersize=5.5, linewidth=2.6)
        plt.plot(muVals, sim_OPT, label='Sim. / OPT',
                 color='blue', marker='o', markersize=5.5, linewidth=2.6)

        # for i in range(8):
        #     plt.scatter(np.arange(i * 100 + 1 + 20 * i * (i > 0), (i + 1) * 100 + 1 + 20 * i * (i > 0)),
        #                 grouped['beta_0' + str(i + 1)], s=12, label=labels[str(i)])
        plt.xlabel(r"$\mu$", fontsize=20)
        plt.ylabel('Ratio', fontsize=20)
        plt.xticks(fontsize=16)  # np.array([.97, 1, 1.03]),
        plt.yticks(np.array([0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]), fontsize=16)
        plt.ylim(bottom=0.93, top=1.005)
        # plt.ylim(bottom=0.96, top=0.99)
        # plt.gca().axes.get_xaxis().set_ticks([])
        # if divideBy == 9:
        #     plt.yticks(np.arange(0.9995, 1.0001, 0.0001))
        #     plt.ylim(bottom=0.9994)
        # elif divideBy == 7:
        #     plt.yticks(np.arange(0.9975, 1.0001, 0.0005))
        # plt.ylim(bottom=0.9994)
        # plt.title('Ratio of Worst LME Reward to OPT Reward', fontsize=13)
        # title = 'ratiosGroupedFor' + str(divideBy) + '.eps'
        plt.legend(loc="lower right", prop={'size': 16})  # , bbox_to_anchor=(1, 1))

        plt.savefig('acrossMus.eps', format='eps', bbox_inches='tight')
        # plt.show()
        plt.cla()