import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.many import DTLZ2
from pymoo.problems.multi import ZDT2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
from nondom import *
from msmops import *

NGEN = 1000


class Problem1(Problem):
    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=2,
            xl=np.array([-2,-2]),
            xu=np.array([2,2]),
        )
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:,0]**2 + X[:,1]**2
        f2 = (X[:,0]-1)**2 + X[:,1]**2
        g1 = 2*(X[:,0]-0.1) * (X[:,0]-0.9) / 0.18
        g2 = - 20*(X[:,0]-0.4) * (X[:,0]-0.6) / 4.8
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])





def main():

    ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)

    algorithm = NSGA2(
        pop_size=100,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True

    )
    for i in range(1):
        problem = four_bar_truss()
        res = minimize(problem,
                       algorithm,
                       ('n_gen', NGEN),
                       seed=0,
                       verbose=False)

        population = res.pop

        # 各個体の目的関数値
        objectives = population.get("F")
        # 各個体の変数値
        X = population.get("X")
        # 各個体の制約違反を取得（制約違反がない場合は0）
        constraints = population.get("G")
        # 各個体の選択状態を取得（Trueが選択された個体）
        is_selected = population.get("feasible")

        # 各個体の選択状態や目的関数値を表示
        print(objectives)
        nds = non_dominated(objectives)
        print(objectives.shape)
        title = f"Standard NSGA-II seed={i}"
        nondom_f1 = [item[0] for item in nds]  # 左側の要素のリスト
        nondom_f2 = [item[1] for item in nds]  # 右側の要素のリスト
        plot_scatter(objectives[:,0], objectives[:,1], title)




main()
