import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.many import DTLZ2
from pymoo.problems.multi import ZDT2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from nondom import *
from pymoo.core.problem import Problem
from scipy.stats import rankdata
from MOEAD_Rank import *

NGEN = 200


class Problem1(Problem):
    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=2,
            xl=np.array([-2, -2]),
            xu=np.array([2, 2]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[:, 0] ** 2 + x[:, 1] ** 2)
        f2 = (x[:, 0] - 1) ** 2 + x[:, 1] ** 2
        g1 = 2 * (x[:, 0] - 0.1) * (x[:, 0] - 0.9) / 0.18
        g2 = - 20 * (x[:, 0] - 0.4) * (x[:, 0] - 0.6) / 4.8
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


# カスタムコールバック関数を定義
def my_callback(algorithm):
    if algorithm.n_gen % 1000 == 0:
        print(f"Generation {algorithm.n_gen}:")
        # best_obj = algorithm.pop.get("F").min()
        # print(f"Best Objective Value: {best_obj:.6f}")
        # print(algorithm.pop.get("F"))


def main():
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)


    for i in range(100):
        print(i)
        algorithm = MOEAD_Rank(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            # callback=my_callback
        )
        problem = get_problem("dtlz2")
        res = minimize(problem,
                       algorithm,
                       ('n_gen', NGEN),
                       seed=i,
                       verbose=False)

        # res は minimize() 関数の実行結果
        population = res.pop

        # 各個体の目的関数値
        objectives1 = population.get("F")
        # 各個体の変数値
        # X = population.get("X")

        # 各個体の選択状態や目的関数値を表示
        #nds = non_dominated(objectives1)
        #print(len(nds))

        # plot_free(F[:, 0], F[:, 1], title=title)
        # plot_free(nondom_f1, nondom_f2, title=title)

        algorithm = NSGA2(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            # callback=my_callback
        )
        res = minimize(problem,
                       algorithm,
                       ('n_gen', NGEN),
                       seed=i,
                       verbose=False)
        population = res.pop
        objectives2 = population.get("F")

        #nds2 = non_dominated(objectives2)
        #print(len(nds2))
        # 各個体の目的関数値
        # 各個体の変数値
        # X = population.get("X")
        plt.title(f"seed={i}")
        x1 = objectives1[:, 0]
        y1 = objectives1[:, 1]
        z1 = objectives1[:,2],
        x2 = objectives2[:, 0]
        y2 = objectives2[:, 1]
        z2 = objectives2[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.invert_yaxis()
        ax.scatter(x1, y1, z1, label="Rank", c="green")
        ax.scatter(x2, y2, z2, label="MOEA/D", c="red", alpha=0.5)
        plt.legend()
        plt.show()

        #Scatter().add(res.F).show()


main()
