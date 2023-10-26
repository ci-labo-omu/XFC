import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.optimize import minimize

from pymoo.util.ref_dirs import get_reference_directions
import matplotlib.pyplot as plt
from nondom import *
import random
import numpy as np
from Scenarios import *
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt
from FreshController5 import NewController5
from NSGA_RANK import *

NGEN = 100

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


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6,
                         n_obj=2,
                         n_constr=0,
                         xl=np.zeros(6),
                         xu=np.ones(6))

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = self.run_2out(x, best_out)
        # out["G"] = np.column_stack([x[:, 0] + x[:, 1] - 1, x[:, 0] - x[:, 1] + 1])

    def feasible(self):
        x1, x2, x3, x4, x5, x6 = self
        return x1 < x2 < x3 < x4 < x5 and -0.2 < x1 < 1.0 and -0.2 < x2 < 1.0 and -0.2 < x3 < 1.0 and -0.2 < x4 < 1.0 and -0.2 < x5 < 1.0 and 0 <= x6 <= 1.0

    # これは制約関数で，制約を満たす場合のみ評価関数を実行する
    def constrained_fitness(self):
        if self.feasible():
            self.f1, self.f2 = self.run_2out(self, best_out)
            return self.f1, self.f2
        else:
            return 0.0, 0.0

    def run_2out(self, gene, gene_out, Scenarioset=Scenario3):
        accu, time = 0, 0
        for scene in Scenarioset:
            f1, f2 = running(gene, gene_out, scene)
            accu += f1
            time += f2
        accu /= len(Scenarioset)
        time /= len(Scenarioset)
        return [accu, time]


def my_callback(algorithm):
    print(f"Rank Generation {algorithm.n_gen}:")
    # best_obj = algorithm.pop.get("F").min()
    k = 0
    if k:
        F = algorithm.pop.get("F")
        f1_rank = rankdata(F[:, 0], method='ordinal')
        f2_rank = rankdata(F[:, 1], method='ordinal')
        plt.figure()
        plt.title(f"Rank, GEN = {algorithm.n_gen}")
        plt.scatter(f1_rank, f2_rank)
        plt.show()
        plt.figure()
        plt.title(f"Objective, GEN = {algorithm.n_gen}")
        plt.scatter(F[:, 0], F[:, 1])
        plt.show()


def main():
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)

    problem = MyProblem()

    algorithm = NSGA2_Rank(
        pop_size=100,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True

    )
    res = minimize(problem,
                   algorithm,
                   ('n_gen', NGEN),
                   verbose=False,
                   callback=my_callback
                   )

    population = res.pop

    # 各個体の目的関数値
    objectives1 = population.get("F")
    # 各個体の変数値
    X = population.get("X")
    print("Scenario 3")
    print("Rank:")
    print(X)

    algorithm = NSGA2(
        pop_size=100,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True

    )
    res = minimize(problem,
                   algorithm,
                   ('n_gen', NGEN),
                   verbose=False)

    population = res.pop

    # 各個体の目的関数値
    objectives2 = population.get("F")
    # 各個体の変数値
    X = population.get("X")
    print("NSGA-II:")
    print(X)

    plt.title("Scenario3 NGEN=100")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(objectives1[:, 0], objectives1[:, 1], label="NSGA-II Rank", c="Green")
    plt.scatter(objectives2[:, 0], objectives2[:, 1], label="NSGA-II", c="red", alpha=0.3)

    plt.legend()
    plt.show()


main()
