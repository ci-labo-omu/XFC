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


    # standard = np.array([0, 100, 200, 90])
    # genes = np.tile(standard, (10, 1))
    # genes = (2 * np.random.rand(10, 4) - 1) * np.array([30, 30, 30, 10]) + genes
    output_standard = np.array(
        [-450, -360, 120, -360, -60, 180, 120, 120, 0, 90, 160, 160, 140, 140, 140, 140, 140, 140])
    new_genes = np.tile(output_standard, (10, 1))  # 縦に，standardを積み重ねる
    out_run_scenario3 = [-476.26230498, -374.16227015, 91.14998438, -339.49869418, -65.13170907,
                         188.37775874, 102.19974504, 111.27493103, -30.16147097, 85.87972781,
                         139.16449379, 147.9620688, 160.80626404, 161.90306892, 167.57717483,
                         149.0889584, 172.93617503, 177.46649927, ]
    # 前9要素が前後の推力，後9要素が角速度の絶対値
    new_genes = (2 * (np.random.rand(10, 18) - 1)) * np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10,
                                                               10, 10, 10, 10, 10]) + new_genes
    gene = [-6.73947456,  31.06759868, 170.07489844, 110.53856904]  # Best in CSEXP
    standard = np.array([0, 100, 200, 90])
    genes = np.tile(gene, (10, 1))
    genes = (2 * np.random.rand(10, 4) - 1) * np.array([30, 30, 30, 10]) + genes

    accu = [run_out_lists(gene, new_genes[j]) for j in range(len(new_genes))]
    first = np.argmax(accu)
    print(first)
    outgene_first = new_genes[first]
    gene_first = genes[first]

    for i in range(3000):
        if i%5 == 0: #5回に1回出力進化
            gene_a, gene_b = tournament(new_genes, accu)
            children = blend(gene_a, gene_b, 0.5)
            score_child = run_out_lists(gene_first, children) # メンバシップ関数には現在の最良のものを使う
            if score_child > np.min(accu):
                new_genes[np.argmin(accu)] = children
                accu[np.argmin(accu)] = score_child
        else: # 奇数回目，メンバシップ関数の方の子個体生成
            gene_a, gene_b = tournament(genes, accu)
            children = blend(gene_a, gene_b, 0.5)
            if children[0] > children[1] or children[1] > children[2]:
                i -= 1
                continue
            score_child = run_out_lists(children, outgene_first)# 出力には現在の最良のものを使う
            if score_child > np.min(accu):
                genes[np.argmin(accu)] = children
                accu[np.argmin(accu)] = score_child

        first = np.argmax(accu)
        outgene_first = new_genes[first]
        gene_first = genes[first]
        print(i+1)
        print(outgene_first)
        print(gene_first)
        with open("Official_both5skip.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accu)
