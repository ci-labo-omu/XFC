import numpy as np


class four_bar_truss:

    def __init__(self):
        # Four bar truss problem
        self.L = 200
        self.F = 10
        self.E = 2 * 10 ** 5
        self.sigma = 10

    def evaluate(self, individual, scenario):
        x1, x2, x3, x4 = individual
        # f1 is same in each scenario
        f1 = self.L * (2 * x1 + np.sqrt(2) * x2 + np.sqrt(2) * x3 + x4)
        if scenario == 1:
            f2 = self.F * self.L / self.E * (2 / x1 + 2 * np.sqrt(2) / x2 - 2 * np.sqrt(2) / x3 + 2 / x4)
        if scenario == 2:
            f2 = self.F * self.L / self.E * (2 / x1 + 2 * np.sqrt(2) / x2 + 4 * np.sqrt(2) / x3 + 2 / x4)
        if scenario == 3:
            f2 = self.F * self.L / self.E * (6 * np.sqrt(2) / x3 + 3 / x4)
        g1, g2, g3, g4 = self.constraint(individual)
        return f1 + 1e5 * (g1 + g2 + g3 + g4), f2 + 1e5 * (g1 + g2 + g3 + g4)

    # constraint
    def constraint(self, individual):
        x1, x2, x3, x4 = individual
        g1 = max(self.F / self.sigma - x1, x1 - 3 * self.F / self.sigma, 0)
        g4 = max(self.F / self.sigma - x4, x4 - 3 * self.F / self.sigma, 0)
        g2 = max(np.sqrt(2) * self.F / self.sigma - x2, x2 - 3 * self.F / self.sigma, 0)
        g3 = max(np.sqrt(2) * self.F / self.sigma - x3, x3 - 3 * self.F / self.sigma, 0)
        return g1, g2, g3, g4


class WeldedBeamProblem:
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