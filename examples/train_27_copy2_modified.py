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
pm = 1/54

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
        controllers = [NewController2(gene, genes2), NewController2(gene, genes2)]
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


    def real_mutation(u):
        p = np.random.rand(len(u))
        p = p <= pm
        u[p] (2 * np.random.rand(10, 7) - 1) * np.array([30, 30, 30, 10, 50, 50, 50]) + u[p]
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


    speeds = np.arange(-450, 450, 100 / 3)
    angles = np.arange(0, 180, 20 / 3)
    output_standard = np.r_[speeds, angles]
    speed_shake = np.full(27, 20)
    angle_shake = np.full(27, 10)
    output_shake = np.r_[speed_shake, angle_shake]

    # new_genes = np.tile(output_standard, (10, 1))  # 縦に，standardを積み重ねる
    out_27 = np.array([0, 90])
    out_27 = np.tile(out_27, (10, 1))
    ss = np.tile(np.array([20, 10]), (10, 1))
    # 前27要素が前後の推力，後27要素が角速度の絶対値
    mod_new_genes = np.array([-461.79048699, -411.80490733, -403.21941771, -367.36119148, -326.08121004,
                     -314.47899696, -262.25148176, -236.41099967, -199.9587384, -168.34704605,
                     -138.48811604, -103.80482541, -92.25424265, -19.2606112, 2.48975399,
                     18.45341191, 56.47548647, 80.31332275, 137.77944239, 139.84951113,
                     203.92632044, 230.02538834, 257.93673747, 292.96312339, 338.59216717,
                     349.37065486, 369.39961917, -15.4698586, -0.94523055, 5.00745265,
                     6.79153201, 22.12231563, 25.79638824, 32.64151245, 41.00649518,
                     41.25269494, 46.64794758, 59.18591287, 55.50019091, 81.79089422,
                     71.7219728, 80.67820036, 84.57853013, 87.396528, 100.67884561,
                     103.40142312, 116.31054855, 125.7682513, 128.92351688, 128.24002384,
                     139.17607817, 158.04437515, 162.12156851, 168.49338955])
    new_genes = (2 * (np.random.rand(10, 54) - 1)) * output_shake + mod_new_genes

    gene = [10.77823972, 31.59573362, 194.69036187, 103.96850551, -20.75314633,
            105.53213439, 125.61431108]  # Best in CSEXP, ここを改良した
    # スピードの三要素をほぼほぼ追加できた，あとはFreShController2の方，スピードを取り出す関数を作る
    genes = np.tile(gene, (10, 1))
    genes = (2 * np.random.rand(10, 7) - 1) * np.array([30, 30, 30, 10, 50, 50, 50]) + genes

    accu = [run_out_lists(gene, new_genes[j]) for j in range(len(new_genes))]
    first = np.argmax(accu)
    print(first)
    outgene_first = new_genes[first]
    gene_first = genes[first]

    for i in range(2000):
        if i % 2:  # 偶数回目，出力の方の子個体生成
            gene_a, gene_b = tournament(new_genes, accu)
            children_out = blend(gene_a, gene_b, 0.5)
            score_child = run_out_lists(gene_first, children_out)  # メンバシップ関数には現在の最良のものを使う
            if score_child > np.min(accu):
                new_genes[np.argmin(accu)] = children_out
                accu[np.argmin(accu)] = score_child
        else:  # 奇数回目，メンバシップ関数の方の子個体生成
            gene_a, gene_b = tournament(genes, accu)
            children = blend(gene_a[4:7], gene_b[4:7], 0.5)
            p = gene.copy()
            p[4:7] = children
            children = real_mutation(children)
            if children[0] > children[1] or children[1] > children[2]:
                i -= 1
                continue
            score_child = run_out_lists(p, outgene_first)  # 出力には現在の最良のものを使う
            if score_child > np.min(accu):
                genes[np.argmin(accu)][4:7] = children
                accu[np.argmin(accu)] = score_child

        first = np.argmax(accu)
        outgene_first = new_genes[first]
        gene_first = genes[first]
        print(i + 1)
        print(outgene_first)
        print(gene_first)
        with open("both_27_speed_only_modified.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accu)
