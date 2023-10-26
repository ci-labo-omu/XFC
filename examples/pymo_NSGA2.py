import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.many import DTLZ2, WFG1, WFG2
from pymoo.problems.multi import ZDT2
from pymoo.util.ref_dirs import get_reference_directions
from nondom import *
from pymoo.core.problem import Problem
from scipy.stats import rankdata
from NSGA_RANK import *

NGEN = 10000


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
        #F = algorithm.pop.get("F")
        #f1_rank = rankdata(F[:, 0], method='ordinal')
        #f2_rank = rankdata(F[:, 1], method='ordinal')
        #plt.figure()
        #plt.title(f"Rank, GEN = {algorithm.n_gen}")
        #plt.scatter(f1_rank, f2_rank)
        #plt.show()
        #plt.figure()
        #plt.title(f"Objective, GEN = {algorithm.n_gen}")
        #plt.scatter(F[:, 0], F[:, 1])
        #plt.show()
        # print(f"Best Objective Value: {best_obj:.6f}")
        # print(algorithm.pop.get("F"))


def main():
    #ref_dirs = get_reference_directions("uniform", 2, n_partitions=12

    peka = DTLZ2().pareto_front()
    for i in range(10):
        algorithm = NSGA2_Rank(
            pop_size=100,
            #ref_dirs=ref_dirs,
            eliminate_duplicates=True,
            callback=my_callback
        )

        print(i)
        problem = DTLZ2()
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
        #X = population.get("X")

        # 各個体の選択状態や目的関数値を表示
        nds = non_dominated(objectives1)


        # plot_free(F[:, 0], F[:, 1], title=title)
        # plot_free(nondom_f1, nondom_f2, title=title)
        algorithm = NSGA2(
            pop_size=100,
            eliminate_duplicates=True,
            #callback=my_callback
        )
        res = minimize(problem,
                       algorithm,
                       ('n_gen', NGEN),
                       seed=i,
                       verbose=False)
        population = res.pop

        # 各個体の目的関数値
        objectives2 = population.get("F")
        # 各個体の変数値
        X = population.get("X")
        #2次元
        #plt.title(f"WFG1  seed={i}")
        plt.scatter(objectives1[:, 0], objectives1[:, 1], label="NSGA-II Rank",c = "Green")
        plt.scatter(objectives2[:, 0], objectives2[:, 1], label="NSGA-II", c = "red", alpha = 0.3)
        #plt.legend()
        x1 = objectives1[:, 0]
        y1 = objectives1[:, 1]
        z1 = objectives1[:,2],
        x2 = objectives2[:, 0]
        y2 = objectives2[:, 1]
        z2 = objectives2[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.invert_yaxis()
        plt.title(f"DTLZ2 seed={i}")
        ax.scatter(peka[:, 0], peka[:, 1], peka[:, 2], label="Pareto front")
        ax.scatter(x1, y1, z1, label="Rank", c="green")
        ax.scatter(x2, y2, z2, label="NSGA-II", c="red", alpha=0.5)
        plt.legend()
        plt.show()


main()
