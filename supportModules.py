import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


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