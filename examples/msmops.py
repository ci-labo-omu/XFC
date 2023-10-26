import os

import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions

from NSGA_RANK import *
from MOEAD_Rank import *
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from nondom import *
from pymoo.core.problem import Problem
from Multi_series import *
from new_file import *


class four_bar_truss(Problem):

    def __init__(self, scenario, ):
        # Four bar truss problem
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=4, xl=np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                         xu=np.array([1000, 1000, 1000, 1000]), )
        self.L = 200
        self.F = 10
        self.E = 2 * 10 ** 5
        self.sigma = 10
        self.scenario = scenario

    def _evaluate(self, x, out, *args, **kwargs):
        # f1 is same in each scenario
        f1 = self.L * (2 * x[:, 0] + np.sqrt(2) * x[:, 1] + np.sqrt(2) * x[:, 2] + x[:, 3])
        if self.scenario == 1:
            f2 = self.F * self.L / self.E * (
                    2 / x[:, 0] + 2 * np.sqrt(2) / x[:, 1] - 2 * np.sqrt(2) / x[:, 2] + 2 / x[:, 3])
        if self.scenario == 2:
            f2 = self.F * self.L / self.E * (
                    2 / x[:, 0] + 2 * np.sqrt(2) / x[:, 1] + 4 * np.sqrt(2) / x[:, 2] + 2 / x[:, 3])
        if self.scenario == 3:
            f2 = self.F * self.L / self.E * (6 * np.sqrt(2) / x[:, 2] + 3 / x[:, 3])
        g1 = np.maximum(self.F / self.sigma - x[:, 0], np.maximum(x[:, 0] - 3 * self.F / self.sigma, 0))
        g4 = np.maximum(self.F / self.sigma - x[:, 3], np.maximum(x[:, 3] - 3 * self.F / self.sigma, 0))
        g2 = np.maximum(np.sqrt(2) * self.F / self.sigma - x[:, 1], np.maximum(x[:, 1] - 3 * self.F / self.sigma, 0))
        g3 = np.maximum(np.sqrt(2) * self.F / self.sigma - x[:, 2], np.maximum(x[:, 2] - 3 * self.F / self.sigma, 0))
        constraint_violation = g1 + g2 + g3 + g4

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4])


class four_bar_truss2(Problem):

    def __init__(self, scenario, ):
        # Four bar truss problem
        super().__init__(n_var=4, n_obj=2, n_constr=0, xl=np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                         xu=np.array([1000, 1000, 1000, 1000]), )
        self.L = 200
        self.F = 10
        self.E = 2 * 10 ** 5
        self.sigma = 10
        self.scenario = scenario

    def _evaluate(self, x, out, *args, **kwargs):
        # 目的関数値の計算
        # f1 is same in each scenario
        f1 = self.L * (2 * x[:, 0] + np.sqrt(2) * x[:, 1] + np.sqrt(2) * x[:, 2] + x[:, 3])
        if self.scenario == 1:
            f2 = self.F * self.L / self.E * (
                    2 / x[:, 0] + 2 * np.sqrt(2) / x[:, 1] - 2 * np.sqrt(2) / x[:, 2] + 2 / x[:, 3])
        if self.scenario == 2:
            f2 = self.F * self.L / self.E * (
                    2 / x[:, 0] + 2 * np.sqrt(2) / x[:, 1] + 4 * np.sqrt(2) / x[:, 2] + 2 / x[:, 3])
        if self.scenario == 3:
            f2 = self.F * self.L / self.E * (6 * np.sqrt(2) / x[:, 2] + 3 / x[:, 3])

        # 制約条件の評価
        g1 = np.maximum(self.F / self.sigma - x[:, 0], np.maximum(x[:, 0] - 3 * self.F / self.sigma, 0))
        g4 = np.maximum(self.F / self.sigma - x[:, 3], np.maximum(x[:, 3] - 3 * self.F / self.sigma, 0))
        g2 = np.maximum(np.sqrt(2) * self.F / self.sigma - x[:, 1], np.maximum(x[:, 1] - 3 * self.F / self.sigma, 0))
        g3 = np.maximum(np.sqrt(2) * self.F / self.sigma - x[:, 2], np.maximum(x[:, 2] - 3 * self.F / self.sigma, 0))
        # 制約違反の合計を計算
        constraint_violation = g1 + g2 + g3 + g4
        print(f1.shape)

        # 制約違反を目的関数値に追加（ペナルティ法など）
        penalty = 1  # 違反した場合のペナルティ
        out["F"] = np.column_stack([f1, f2 + penalty * constraint_violation])


class WeldedBeamProblem(Problem):
    def __init__(self):
        self.rho = 0.284  # density lb/in^3
        self.F = 6000  # mass lb
        self.E = 29 * 10 ** 6  # modulus of elasticity psi
        self.G = 11.5 * 10 ** 6  # shear modulus psi
        self.T = 25000  # torque lb/in
        self.sigma = 30000  # stress limit psi
        self.k = 0.87  # constant parameter
        self.phi = 13600  # yield stress psi

    def evaluate(self, individual, scenario):
        x1, x2, x3 = individual
        if scenario == 1:
            # f1 and f2 in scenario 1
            f1_1 = self.rho * x1 * x2 * x3
            f2_1 = 2 * self.F ** 2 * x3 ** 3 / (self.E * x1 * x2 ** 3)
            g1_violation = self.constraint_g1(x1, x2, x3)
            return (f1_1 + f2_1) + 1e5 * g1_violation,

        if scenario == 2:
            # f1 and f2 in scenario 2
            f1_2 = self.rho * x1 * x2 * x3
            f2_2 = self.T ** 2 * x3 / (2 * self.G * max(x1, x2) * min(x1, x2) ** 3)
            g2_violation = self.constraint_g2(x1, x2, x3)
            return (f1_2 + f2_2) + 1e5 * g2_violation,

    def constraint_g1(self, x1, x2, x3):
        g1 = 6 * self.F * x3 / (x2 * x1 ** 2) - self.sigma
        return max(0, g1)  # 制約違反の量を返す

    def constraint_g2(self, x1, x2, x3):
        g2 = self.k * self.T / (max(x1, x2) * min(x1, x2) ** 2) - self.phi
        return max(0, g2)  # 制約違反の量を返す


def main1():
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=100)

    problem = four_bar_truss(2)
    algorithm1 = NSGA2_Rank(
        #pop_size=150,
         ref_dirs=ref_dirs,

        # eliminate_duplicates=True,
    )
    algorithm2 = NSGA2(
        pop_size=150,
        # ref_dirs=ref_dirs,
        # eliminate_duplicates=True,
    )

    res1 = minimize(problem,
                    algorithm1,
                    ('n_gen', 2000),
                    verbose=False)

    res2 = minimize(problem,
                    algorithm2,
                    ('n_gen', 2000),
                    verbose=False)

    # res は minimize() 関数の実行結果
    population1 = res1.pop
    population2 = res2.pop

    # 各個体の目的関数値
    objectives1 = population1.get("F")
    objectives2 = population2.get("F")
    # 各個体の変数値
    X1 = population1.get("X")
    X2 = population2.get("X")

    # print(objectives1)
    print(objectives2)
    # print(f"X1 = {X1}")
    # print(f"X2 = {X2}")
    x1 = objectives1[:, 0]
    y1 = objectives1[:, 1]
    x2 = objectives2[:, 0]
    y2 = objectives2[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim(0, 3500)
    plt.ylim(0, 0.10)
    plt.title("Four Bar Truss Problem Scenario3")
    ax.scatter(x1, y1, label='NSGA-II Rank')
    ax.scatter(x2, y2, label='NSGA-II', alpha=0.3)
    plt.legend()
    plt.show()
    # with open('newnew.py', 'a') as file:
    #    file.write(f"\n#Scenario 1\n")
    #    file.write(f"X1 = {list(X1)}\n")
    #    file.write(f"X2 = {list(X2)}\n")
    #    file.write(f"objectives1 = {list(objectives1)}\n")
    #    file.write(f"objectives2 = {list(objectives2)}\n")


def main2():
    X_Ranks = [X2_Rankseries123, X2_Rankseries231, X2_Rankseries312]
    Xs = [X2_series123, X2_series231, X2_series312]
    X_ref = [X_1, X_2, X_3]
    X_Ranks1 = [X_Rank_1, X_Rank_2, X_Rank_3]
    F_ref = [F_1, F_2, F_3]
    for i in range(1, 4):
        for x_rank, x in zip(X_Ranks, Xs):
            res1 = four_bar_truss(i).evaluate(np.array(x_rank))
            res2 = four_bar_truss(i).evaluate(np.array(x))
            res3 = four_bar_truss(i).evaluate(np.array(X_1))
            res4 = four_bar_truss(i).evaluate(np.array(X_2))
            res5 = four_bar_truss(i).evaluate(np.array(X_3))

            objectives1 = res1[0]
            objectives2 = res2[0]
            x1 = objectives1[:, 0]
            y1 = objectives1[:, 1]
            x2 = objectives2[:, 0]
            y2 = objectives2[:, 1]
            x3 = np.array(res3[0])[:, 0]
            y3 = np.array(res3[0])[:, 1]
            x4 = np.array(res4[0])[:, 0]
            y4 = np.array(res4[0])[:, 1]
            x5 = np.array(res5[0])[:, 0]
            y5 = np.array(res5[0])[:, 1]
            fig = plt.figure()
            ax = fig.add_subplot()
            print(id(x_rank))
            plt.xlim(0, 3500)
            plt.ylim(0, 0.1101)
            plt.yticks(np.arange(0, 0.111, 0.01))

            plt.title(f"Four Bar Truss Problem Scenario{i}")
            ax.scatter(x1, y1, label='NSGA-II Rank')
            ax.scatter(x2, y2, label='NSGA-II', alpha=0.3)
            ax.scatter(x3, y3, label='NSGA-II-SC1', alpha=0.3)
            ax.scatter(x4, y4, label='NSGA-II-SC2', alpha=0.3)
            ax.scatter(x5, y5, label='NSGA-II-SC3', alpha=0.3)
            plt.legend(loc='lower left')
            filename = f'series{id(x_rank)}-{i}.png'
            directory = 'Multi2'  # 'AA'フォルダに保存
            folder_path = os.path.join(os.getcwd(), directory)
            file_path = os.path.join(folder_path, filename)
            plt.savefig(file_path)
            plt.show()


def main3():
    problem1 = four_bar_truss(1)
    problem2 = four_bar_truss(3)
    problem3 = four_bar_truss(2)

    algorithm1 = NSGA2_Rank(
        pop_size=150,
        # ref_dirs=ref_dirs,
        eliminate_duplicates=True,
    )
    algorithm2 = NSGA2(
        pop_size=150,
        # ref_dirs=ref_dirs,
        eliminate_duplicates=True,
    )
    res1_1 = minimize(problem1,
                      algorithm1,
                      ('n_gen', 2000),
                      verbose=False)
    res1_2 = minimize(problem2,
                      algorithm1,
                      ('n_gen', 2000),
                      verbose=False,
                      pop_init=res1_1.pop)
    res1 = minimize(problem3,
                    algorithm1,
                    ('n_gen', 2000),
                    verbose=False,
                    pop_init=res1_2.pop)

    res2_1 = minimize(problem1,
                      algorithm2,
                      ('n_gen', 2000),
                      verbose=False)
    res2_2 = minimize(problem2,
                      algorithm2,
                      ('n_gen', 2000),
                      verbose=False,
                      pop_init=res2_1.pop)
    res2 = minimize(problem3,
                    algorithm2,
                    ('n_gen', 2000),
                    verbose=False,
                    pop_init=res2_2.pop)

    # res は minimize() 関数の実行結果
    population1 = res1.pop
    population2 = res2.pop
    # 各個体の目的関数値
    objectives1 = population1.get("F")
    objectives2 = population2.get("F")
    # 各個体の変数値
    X1 = population1.get("X")
    X2 = population2.get("X")
    print(f"X1 = {X1}")
    print(f"X2 = {X2}")

    # np.savez('X_Series2.npz', X_Rank_123=XR123, X_123=X123, X_Rank_132=X1, X_132=X2)

    # with open('Multi_series.py', 'a') as file:
    #   file.write(f"\n#Learning 1->3->2 \n")
    #   file.write(f"X_Rank132 = {list(X1)}\n")
    #   file.write(f"X_132 = {list(X2)}\n")
    #   file.write(f"F_Rank132 = {list(objectives1)}\n")
    #   file.write(f"F_132 = {list(objectives2)}\n")


class four_bar_truss3(Problem):

    def __init__(self, ):
        # Four bar truss problem
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=4, xl=np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                         xu=np.array([1000, 1000, 1000, 1000]), )
        self.L = 200
        self.F = 10
        self.E = 2 * 10 ** 5
        self.sigma = 10
        self.counter = 0

    def increment(self):
        self.counter += 1
        return self.counter

    def _evaluate(self, x, out, *args, **kwargs):
        print("あ")
        # f1 is same in each scenario
        gen = self.increment()
        f1 = self.L * (2 * x[:, 0] + np.sqrt(2) * x[:, 1] + np.sqrt(2) * x[:, 2] + x[:, 3])
        if gen % 3 == 1:
            print(f"{gen} 1")
            f2 = self.F * self.L / self.E * (
                    2 / x[:, 0] + 2 * np.sqrt(2) / x[:, 1] - 2 * np.sqrt(2) / x[:, 2] + 2 / x[:, 3])
        if gen % 3 == 2:
            print(f"{gen} 2")

            f2 = self.F * self.L / self.E * (
                    2 / x[:, 0] + 2 * np.sqrt(2) / x[:, 1] + 4 * np.sqrt(2) / x[:, 2] + 2 / x[:, 3])
        if gen % 3 == 0:
            print(f"{gen} 3")
            f2 = self.F * self.L / self.E * (6 * np.sqrt(2) / x[:, 2] + 3 / x[:, 3])
        g1 = np.maximum(self.F / self.sigma - x[:, 0], np.maximum(x[:, 0] - 3 * self.F / self.sigma, 0))
        g4 = np.maximum(self.F / self.sigma - x[:, 3], np.maximum(x[:, 3] - 3 * self.F / self.sigma, 0))
        g2 = np.maximum(np.sqrt(2) * self.F / self.sigma - x[:, 1], np.maximum(x[:, 1] - 3 * self.F / self.sigma, 0))
        g3 = np.maximum(np.sqrt(2) * self.F / self.sigma - x[:, 2], np.maximum(x[:, 2] - 3 * self.F / self.sigma, 0))
        constraint_violation = g1 + g2 + g3 + g4

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4])


def main4():
    algorithm1 = NSGA2_Rank(
        pop_size=150,
        # ref_dirs=ref_dirs,
        eliminate_duplicates=True,
    )
    algorithm2 = NSGA2(
        pop_size=150,
        # ref_dirs=ref_dirs,
        eliminate_duplicates=True,
    )
    res1 = minimize(four_bar_truss3(),
                    algorithm1,
                    ('n_gen', 201),
                    verbose=False)

    res2 = minimize(four_bar_truss3(),
                    algorithm2,
                    ('n_gen', 201),
                    verbose=False, )

    # res は minimize() 関数の実行結果
    population1 = res1.pop
    population2 = res2.pop
    # 各個体の目的関数値
    objectives1 = population1.get("F")
    objectives2 = population2.get("F")
    # 各個体の変数値
    X1 = population1.get("X")
    X2 = population2.get("X")
    print(f"X1 = {X1}")
    print(f"X2 = {X2}")
    x1 = objectives1[:, 0]
    y1 = objectives1[:, 1]
    x2 = objectives2[:, 0]
    y2 = objectives2[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim(0, 3500)

    plt.title("Four Bar Truss Problem Scenario3")
    ax.scatter(x1, y1, label='NSGA-II Rank')
    ax.scatter(x2, y2, label='NSGA-II', alpha=0.3)
    plt.legend()
    plt.show()

    # np.savez('X_Series2.npz', X_Rank_123=XR123, X_123=X123, X_Rank_132=X1, X_132=X2)

    with open('Multi_series.py', 'a') as file:
        file.write(f"\n#Learning 2->3->1 * 2000 \n")
        file.write(f"X_Rank111 = {list(X1)}\n")
        file.write(f"X111 = {list(X2)}\n")
        #file.write(f"F_Rank = {list(objectives1)}\n")
        #file.write(f"F = {list(objectives2)}\n")


main1()
