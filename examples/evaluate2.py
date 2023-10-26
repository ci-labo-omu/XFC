import math
import random

from matplotlib import pyplot as plt

from FreshController import NewController
from FreshController3 import NewController3
from src.kesslergame import TrainerEnvironment, KesslerController
from src.kesslergame import Scenario
from Scenarios import *
from Scenarios_ import *
import numpy as np
from result_train import *

rng = np.random.default_rng()
child = 1

# Instantiate an instance of TrainerEnvironment.
# The default settings should be sufficient, but check out the documentation for more information
settings = {
    "graphics_on": True,
    "sound_on": False,
    # "frequency": 60,
    "real_time_multiplier": 200,
    # "lives": 3,
    # "prints": True,
    "allow_key_presses": False
}
best_out = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
            157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
            105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
            -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
            25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
            75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]


def running(gene, genes2, scenario):
    # f1, f2を正規化せず，正答率はそのままに，時間は，生存した場合短い方が，生存しなかった場合長い方がよい．
    # 生存した場合は1 / time, 生存しなかった場合はtime
    game = TrainerEnvironment()
    # Because of how the arcade library is implemented, there are memory leaks for instantiating the environment
    # too many times, keep instantiations to a small number and simply reuse the environment
    controllers = [NewController5(gene, genes2), NewController5(gene, genes2)]
    pre = time.perf_counter()
    score, perf_data = game.run(scenario=scenario, controllers=controllers)
    reason = (score.stop_reason)

    f1 = 1 - np.average([team.accuracy for team in score.teams])
    f2 = score.sim_time / 120 if reason == reason.no_asteroids else 1 - score.sim_time / 120
    # 最小化問題
    return f1, f2


def evaluate(gene, Scenarioset):
    scores = [running(gene, best_out ,Scenario, ) for Scenario in Scenarioset]
    print(scores)
    return scores
def evaluate_vec(genes, Scenarioset):
    scores = [evaluate(gene, Scenarioset) for gene in genes]
    return scores


def run_2out(gene, gene_out):
    accu, time = 0, 0
    for scene in Scenario3:
        f1, f2 = running(gene, gene_out, scene)
        accu += f1
        time += f2
    accu /= len(Scenario3)
    time /= len(Scenario3)
    print(accu, time)
    return [accu, time]
def main():
    result = result1in3
    result_Rank = result1_Rankin1
    resultf1_1 = result[:, 0, 0]
    resultf2_1 = result[:, 0, 1]
    resultf1_2 = result[:, 1, 0]
    resultf2_2 = result[:, 1, 1]
    resultf1_3 = result[:, 2, 0]
    resultf2_3 = result[:, 2, 1]
    resultf1_1_Rank = result_Rank[:, 0, 0]
    resultf2_1_Rank = result_Rank[:, 0, 1]
    resultf1_2_Rank = result_Rank[:, 1, 0]
    resultf2_2_Rank = result_Rank[:, 1, 1]
    resultf1_3_Rank = result_Rank[:, 2, 0]
    resultf2_3_Rank = result_Rank[:, 2, 1]

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Scenario1")
    plt.scatter(resultf1_3, resultf2_3, label="NSGA-II")
    plt.scatter(resultf1_3_Rank, resultf2_3_Rank, label="Rank")
    plt.legend()
    plt.show()

main()
