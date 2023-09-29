import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.many import DTLZ2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from scipy.stats import rankdata
from nondom import *
from pymo_rank import *

NGEN = 1000

class Problem1(Problem):


    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=2,
            xl=np.array([-2, -2]),
            xu=np.array([2, 2]),
        )
        self.values_f1 = []
        self.values_f2 = []

    def _fomula(self, X, out, *args, **kwargs):
        f1 = X[:, 0] ** 2 + X[:, 1] ** 2
        f2 = (X[:, 0] - 1) ** 2 + X[:, 1] ** 2
        g1 = 2 * (X[:, 0] - 0.1) * (X[:, 0] - 0.9) / 0.18
        g2 = - 20 * (X[:, 0] - 0.4) * (X[:, 0] - 0.6) / 4.8

        return f1, f2, g1, g2

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, g1, g2 = self._fomula(x, out, *args, **kwargs)
        self.values_f1.extend(f1)
        self.values_f2.extend(f2)
        # values_f1，values_f2にf1，f2を追加していく．
        # 与えられたxに対してのみ，f1_rank, f2_rankを計算する．
        # f1_rank, f2_rankは，f1, f2の値が小さいほど，小さい値をとる．
        f1_rank = rankdata(f1, method="ordinal")[-100:]
        f2_rank = rankdata(f2, method="ordinal")[-100:]
        out["F"] = np.column_stack([f1_rank, f2_rank])
        out["G"] = np.column_stack([g1, g2])


# カスタムコールバック関数を定義
def my_callback(algorithm):
    if algorithm.n_gen % 1000 == 0:
        print(f"Generation {algorithm.n_gen}:")
        best_obj = algorithm.pop.get("F").min()
        print(f"Best Objective Value: {best_obj:.6f}")


def main():

    ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)

    algorithm = NSGA2(
        pop_size=100,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True,
        #callback=my_callback
    )
    for i in range(100):
        problem = Problem2()

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 10000),
                       seed=i,
                       verbose=False)

        # res は minimize() 関数の実行結果
        population = res.pop

        # 各個体の目的関数値
        objectives = population.get("F")
        # 各個体の変数値
        X = population.get("X")
        # 各個体の制約違反を取得（制約違反がない場合は0）
        constraints = population.get("G")

        # 各個体の選択状態を取得（Trueが選択された個体）
        is_selected = population.get("feasible")
        f1 = X[:, 0] ** 2 + X[:, 1] ** 2
        f2 = (X[:, 0] - 1) ** 2 + X[:, 1] ** 2
        F = np.column_stack([f1, f2])
        # 各個体の選択状態や目的関数値を表示
        nds = non_dominated(F)
        plt.title(f"Rank Archive method seed={i} (gen=10000)")
        plt.xlabel("f1")
        plt.ylabel("f2")
        print(len(nds))
        nondom_f1 = [item[0] for item in nds]  # 左側の要素のリスト
        nondom_f2 = [item[1] for item in nds]  # 右側の要素のリスト
        plt.scatter(nondom_f1, nondom_f2)
        plt.show()


main()
