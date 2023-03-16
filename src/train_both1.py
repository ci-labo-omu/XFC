import math
import random
from FreshController import NewController
from src.kesslergame import TrainerEnvironment, KesslerController
from src.kesslergame import Scenario
from Scenarios import *
import numpy as np
import csv

# メンバシップ関数と出力を両方進化させる
# メンバシップ関数変更ｰ>評価ｰ>出力ｰ>評価で1世代
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


    def running(gene, genes2, scenario):
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Individual", list, fitness=creator.FitnessMax)
        game = TrainerEnvironment()
        # Because of how the arcade library is implemented, there are memory leaks for instantiating the environment
        # too many times, keep instantiations to a small number and simply reuse the environment
        controllers = [NewController(gene, genes2), NewController(gene, genes2)]
        score, perf_data = game.run(scenario=scenario, controllers=controllers)

        """print('Scenario eval time: ' + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))"""

        return sum([team.asteroids_hit for team in score.teams])


    def run_out_lists(gene, gene_out):
        score = [running(gene, gene_out, scene) for scene in Scenario_list]
        return np.average(score)


    def blend(x_a, x_b, alpha):
        u = []
        for i in range(len(x_a)):
            A = np.minimum(x_a[i], x_b[i]) - alpha * np.fabs(x_a[i] - x_b[i])
            B = np.maximum(x_a[i], x_b[i]) + alpha * np.fabs(x_a[i] - x_b[i])
            u.append(np.random.uniform(A, B))
        # u = np.clip(u, x_min, x_max)
        return u


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

    new_genes = [-489.22075336, -366.91465568, 88.17631445, -382.53532176, -97.0264826,
                 168.01903803, 99.03163254, 123.16561505, - 7.67767851, 81.7559517,
                 149.60633281, 146.73745856, 133.71504947, 136.20653374, 130.00619218,
                 125.16709215, 147.07681298, 120.60473168]
    new_genes = np.tile(new_genes, (10, 1))  # 縦に，standardを積み重ねる
    # 前9要素が前後の推力，後9要素が角速度の絶対値
    new_genes = (2 * (np.random.rand(10, 18) - 1)) * np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10,
                                                               10, 10, 10, 10, 10]) + new_genes
    gene = [-28.5890275, 34.22384198, 187.8046941, 93.70131328] #Official_bothより
    genes = np.tile(gene, (10, 1))
    genes = (2 * np.random.rand(10, 4) - 1) * np.array([30, 30, 30, 10]) + genes


    accu = [run_out_lists(gene, new_genes[j]) for j in range(len(new_genes))]
    first = np.argmax(accu)
    print(first)
    outgene_first = new_genes[first]
    gene_first = genes[first]

    for i in range(10000):
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

        first = np.argmax(accu)
        outgene_first = new_genes[first]
        gene_first = genes[first]
        print(i + 1)
        print(outgene_first)
        print(gene_first)
        with open("Official_both1.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accu)
