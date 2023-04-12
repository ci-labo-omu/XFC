import random
import numpy as np
from Scenarios import *
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt
from func_emo import selNSGA22


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
        controllers = [NewController5(gene, genes2), NewController5(gene, genes2)]
        pre = time.perf_counter()
        score, perf_data = game.run(scenario=scenario, controllers=controllers)
        f1 = np.average([team.accuracy for team in score.teams])
        f2 = 1 - score.sim_time / 120 if sum([team.deaths for team in score.teams]) < 3 else score.sim_time / 120
        # 両方最大化
        return f1, f2


    best_out = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]

    f_list = [(0.0, 0.0) for i in range(50)]


    def run_2out(gene, gene_out):
        accu, time = 0, 0
        for scene in Scenario_list:
            f1, f2 = running(gene, gene_out, scene)
            accu += f1
            time += f2
        accu /= len(Scenario_list)
        time /= len(Scenario_list)

        return [accu, time]


    def feasible(individual1):
        x1, x2, x3, x4, x5, x6 = individual1
        return x1 < x2 < x3 < x4 < x5 and -0.2 < x1 < 1.0 and -0.2 < x2 < 1.0 and -0.2 < x3 < 1.0 and -0.2 < x4 < 1.0 and -0.2 < x5 < 1.0 and 0 <= x6 <= 1.0


    # 制約適応度
    def constrained_fitness(individual):

        if not feasible(individual):
            individual.f1 = 0.0
            individual.f2 = 0.0
            return 0.0, 0.0
        else:
            a = run_2out(individual, best_out)
            individual.f1, individual.f2 = a
            print(a)
            return a[0], a[1]


    NGEN = 100  # 繰り返し世代数
    MU = 200  # 集団内の個体数
    CXPB = 0.9  # 交叉率
    """
    def evaluate_individual_rank(individual):
        f1, f2 = constrained_fitness(individual)
        individual.f1 = f1
        individual.f2 = f2
        f1_rank = sum(f1 <= f[0] for f in f_list)
        f2_rank = sum(f2 <= f[1] for f in f_list)
        return f1_rank, f2_rank
    """

    f1_values = np.zeros((MU, 2))
    f2_values = np.zeros((MU, 2))

    # make ranking from f1_values, f2_values, sorting each value

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin, f1=0.0, f2=0.0)


    class IndividualWithF(creator.Individual):
        def evaluate_individual_rank(self):
            f1_values, f2_values = values_pop(pop)
            self.f1, self.f2 = constrained_fitness(self)
            return sum(f1 < self.f1 for f1 in f1_values), sum(f2 < self.f2 for f2 in f2_values)

        def return_rank(individual, f1_values, f2_values):
            return sum(f1 < individual.f1 for f1 in f1_values), sum(f2 < individual.f2 for f2 in f2_values)


    toolbox = base.Toolbox()
    dim = 6
    toolbox.register("attr_float", random.uniform, -0.2, 1.0)
    toolbox.register("attr_angle", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initCycle, IndividualWithF,
                     (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float,
                      toolbox.attr_float, toolbox.attr_float, toolbox.attr_angle), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", IndividualWithF.evaluate_individual_rank)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-0.2, up=1.0, eta=20.0)

    toolbox.register("mutates", tools.mutPolynomialBounded, up=1, low=-0.2, indpb=1 / 6, eta=30)
    toolbox.register("select", tools.selNSGA2, )  # 自分でまねて，リスト考えて実装

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.average, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"


    def values_pop(pop):
        pp = np.array([[ind.f1, ind.f2] for ind in pop])
        f1_values, f2_values = pp.T[0], pp.T[1]
        return f1_values, f2_values


    # 第一世代の生成
    pop = toolbox.population(n=MU)
    pop_init = pop[:]
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)

    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # 最適計算の実行
    for gen in range(1, NGEN):
        # 子母集団生成
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 交叉と突然変異
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # 交叉させる個体を選択
            if random.random() <= CXPB:
                # 交叉
                toolbox.mate(ind1, ind2)
            # 突然変異
            toolbox.mutates(ind1)
            toolbox.mutates(ind2)
            # 交叉と突然変異させた個体は適応度を削除する
            del ind1.fitness.values, ind2.fitness.values, ind1.f1, ind2.f2
        # 適応度を削除した個体について適応度の再評価を行う
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 次世代を選択
        print(pop)
        pop = toolbox.select(pop + offspring, MU)
        print(f"Generation {gen}:")
        print(logbook.stream)
        for j, ind in enumerate(pop):
            print("Individual ", j + 1, ": ", ind, "Rank: ", ind.fitness.values, "Fitness: ", ind.fitness.values)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

    print(pop, pop_init, stats)
    fitnesses_init = np.array([list(pop_init[i].fitness.values) for i in range(len(pop_init))])
    fitnesses = np.array([list(pop[i].fitness.values) for i in range(len(pop))])
    plt.plot(fitnesses_init[:, 0], fitnesses_init[:, 1], "b.", label="Initial")
    plt.plot(fitnesses[:, 0], fitnesses[:, 1], "r.", label="Optimized")
    plt.legend(loc="upper right")
    plt.title("fitnesses")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.show()
