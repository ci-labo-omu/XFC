import os

from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions
from NSGA_RANK import *
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from nondom import *
from pymoo.core.problem import Problem
from nondom import *


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
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4])
        return out["F"], out["G"]

def main():
    G_max = 10000
    problem = four_bar_truss
    crossover = SBX(prob=0.9)
    mutation = PM(prob=1 / 4, eta=20)
    X = np.random.rand( 100, 4)*10
    print(X)
    P = Population.new("X", X)
    four_bar_truss(1).evaluate(P.get("X"), P)
    print(P.get("X"))

    #or t in range(1, G_max+1):
    #or i in range(3):
    P_c = crossover.do(four_bar_truss(1), P)
    print("ああ")
    print(P_c)
    P_c = mutation.do(four_bar_truss(1), P_c)
    print(P_c)
    C_t = Population.merge(P, P_c)
    print(C_t)

main()

