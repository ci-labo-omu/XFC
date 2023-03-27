import math
import random
from FreshController import NewController
from src.kesslergame import TrainerEnvironment, KesslerController
from src.kesslergame import Scenario
from Scenarios import *
from Scenarios_ import *
import numpy as np
import csv

rng = np.random.default_rng()
child = 1

if __name__ == "__main__":
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
    # 前15個が加速度，後15個が角速度
    out_base = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]


    def running(gene, genes2, scenario):
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Individual", list, fitness=creator.FitnessMax)
        game = TrainerEnvironment()
        # Because of how the arcade library is implemented, there are memory leaks for instantiating the environment
        # too many times, keep instantiations to a small number and simply reuse the environment
        controllers = [NewController3(gene, genes2), NewController3(gene, genes2)]
        score, perf_data = game.run(scenario=scenario, controllers=controllers)

        """print('Scenario eval time: ' + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))"""

        return sum([team.asteroids_hit for team in score.teams])


    def run_collection(gene, out):
        score = [running(gene, out, scene) for scene in Scenario_list]
        return np.average(score)


    def blend(x_a, x_b, alpha):
        u = []
        for i in range(4):
            A = np.minimum(x_a[i], x_b[i]) - alpha * np.fabs(x_a[i] - x_b[i])
            B = np.maximum(x_a[i], x_b[i]) + alpha * np.fabs(x_a[i] - x_b[i])
            u.append(np.random.uniform(A, B))
        # u = np.clip(u, x_min, x_max)
        return u


    def run_out_lists(gene, gene_out):
        score = [running(gene, gene_out, scene) for scene in Scenario_list]
        return np.average(score)


    def tournament(_genes, _accu):
        d, e, f, g = rng.choice(len(_genes), 4, replace=False)
        if _accu[d] > _accu[e]:
            gene_a = _genes[d]
        else:
            gene_a = _genes[e]
        if _accu[f] > _accu[g]:
            gene_b = _genes[f]
        else:
            gene_b = _genes[g]
        return gene_a, gene_b


    new_genes = np.tile(out_base, (10, 1))
    new_genes = (2 * (np.random.rand(10, 30) - 1)) * np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                                                               20, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                                               10, 10, 10, 10, 10, 10]) + new_genes
    standard = np.array([0, 100, 200, 350, 500, 90])
    genes = np.tile(standard, (10, 1))
    genes = (2 * np.random.rand(10, 6) - 1) * np.array([30, 30, 30,30,30, 10]) + genes
    ave_list = []
    accu = [run_out_lists(genes[j], new_genes[j]) for j in range(len(new_genes))]
    first = np.argmax(accu)
    outgene_first = new_genes[first]
    gene_first = genes[first]

    for i in range(2000):
        if i % 2:  # 偶数回目，出力の方の子個体生成
            gene_a, gene_b = tournament(new_genes, accu)
            children = blend(gene_a, gene_b, 0.5)
            score_child = run_out_lists(gene_first, children)  # メンバシップ関数には現在の最良のものを使う
            if score_child > np.min(accu):
                new_genes[np.argmin(accu)] = children
                accu[np.argmin(accu)] = score_child
        else:  # 奇数回目，メンバシップ関数の方の子個体生成
            gene_a, gene_b = tournament(genes, accu)
            children = blend(gene_a, gene_b, 0.5)
            if children[0] > children[1] or children[1] > children[2]:
                i -= 1
                continue
            score_child = run_out_lists(children, outgene_first)  # 出力には現在の最良のものを使う
            if score_child > np.min(accu):
                genes[np.argmin(accu)] = children
                accu[np.argmin(accu)] = score_child
        with open("Both_new_dist.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accu)

        with open("Both_new_dist_gene.csv","a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(genes)
