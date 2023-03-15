import pandas as pd
import matplotlib.pyplot as plt

def plotRatios(betaVal):
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

    # plt.gca().axes.get_yaxis().set_ticks([])
    if divideBy == 9:
        plt.yticks(np.arange(0.9995, 1.0001, 0.0001))
        plt.ylim(bottom=0.9994)
    elif divideBy == 7:
        plt.yticks(np.arange(0.9975, 1.0001, 0.0005))
        # plt.ylim(bottom=0.9994)
    plt.title('Ratio of Worst LME Reward to OPT Reward', fontsize=13)
    title = 'ratiosFor' + str(divideBy) + '.eps'
    plt.legend(loc="lower left", prop={'size': 10}, ncol=2)#, bbox_to_anchor=(1, 1))

    plt.savefig(title, format='eps', bbox_inches='tight')
    # plt.show()
    plt.cla()