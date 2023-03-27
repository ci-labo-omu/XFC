import math
import random
from FreshController import NewController
from kesslergame import TrainerEnvironment, KesslerController
from kesslergame import Scenario
from Scenarios import *
from deap import creator, base, tools, algorithms
import numpy as np
import csv
import matplotlib.pyplot as plt

# both evolve
rng = np.random.default_rng()
child = 1

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
        controllers = [NewController4(gene, genes2), NewController4(gene, genes2)]
        pre = time.perf_counter()
        score, perf_data = game.run(scenario=scenario, controllers=controllers)
        """print('Scenario eval time: ' + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))"""
        f1 = np.average([team.accuracy for team in score.teams])
        f2 = 1 - score.sim_time / 120 if sum([team.deaths for team in score.teams]) < 3 else score.sim_time / 120
        return f1, f2


    # あとは，run_2outから2つのあれを出すだけ
    best_out = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]


    def run_2out(gene, gene_out):
        accu, time = 0, 0
        for scene in Scenario_list:
            f1, f2 = running(gene, gene_out, scene)
            accu += f1
            time += f2
        accu /= len(Scenario_list)
        time /= len(Scenario_list)
        return accu, time


    def feasible(individual1):
        x1, x2, x3, x4, x5, x6 = individual1
        return x1 < x2 < x3 < x4 < x5 and -0.2 < x1 < 1.0 and -0.2 < x2 < 1.0 and -0.2 < x3 < 1.0 and -0.2 < x4 < 1.0 and -0.2 < x5 < 1.0 and 0 <= x6 <= 1.0


    def feasible_out(individual2):
        return abs(np.array(individual2[0:15]).any()) <= 480 and abs(np.array(individual2[16:30]).any()) <= 180


    # 制約適応度
    """
    def constrained_fitness(individual1, individual2):
        if not feasible(individual1) and not feasible_out(individual2):
            return 0, 0
        else:
            return run_2out(individual1, individual2)"""


    def constrained_fitness(best, individual2):
        if not feasible_out(individual2):
            return 0, 0
        else:
            return run_2out(best, individual2)

    def constrained_fitness1(individual1, best_out):
        if not feasible(individual1):
            return 0, 0
        else:
            return run_2out(individual1, best_out)

    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual1", list, fitness=creator.FitnessMax)
    creator.create("Individual2", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    dim = 6
    toolbox.register("attr_float", random.uniform, -0.2, 1.0)
    toolbox.register("attr_angle", random.uniform, 0, 1.0)
    toolbox.register("attr_out_speed", random.uniform, -1.0, 1.0)
    toolbox.register("attr_out_angle", random.uniform, -1.0, 1.0)
    toolbox.register("individual1", tools.initCycle, creator.Individual1,
                     (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float,
                      toolbox.attr_float, toolbox.attr_float, toolbox.attr_angle), n=1)

    toolbox.register("individual2", tools.initCycle, creator.Individual2,
                     (15 * (toolbox.attr_out_speed,) +
                      15 * (toolbox.attr_out_angle,)), n=1)

    toolbox.register("population1", tools.initRepeat, list, toolbox.individual1)
    toolbox.register("population2", tools.initRepeat, list, toolbox.individual2)
    toolbox.register("evaluate1", run_2out)
    toolbox.register("evaluate1", constrained_fitness)
    toolbox.decorate("evaluate1", tools.DeltaPenalty(feasible, 0))
    toolbox.register("evaluate2", run_2out)
    toolbox.register("evaluate2", constrained_fitness)
    toolbox.decorate("evaluate2", tools.DeltaPenalty(feasible_out, 0))

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-0.2, up=1.0, eta=20.0)
    toolbox.register("mate2", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=20.0)

    toolbox.register("mutates1", tools.mutPolynomialBounded, up=1, low=-0.2, indpb=1 / 6, eta=30)
    toolbox.register("mutates2", tools.mutPolynomialBounded, up=1.0, low=-1.0, indpb=1 / 15, eta=30)

    toolbox.register("select", tools.selNSGA2)


    def blend(x_a, x_b, alpha):
        u = []
        for i in range(len(x_a)):
            A = np.minimum(x_a[i], x_b[i]) - alpha * np.fabs(x_a[i] - x_b[i])
            B = np.maximum(x_a[i], x_b[i]) + alpha * np.fabs(x_a[i] - x_b[i])
            u.append(np.random.uniform(A, B))
        # u = np.clip(u, x_min, x_max)
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


    standard = np.array([0, 100, 200, 350, 500, 90])

    NGEN = 100  # 繰り返し世代数
    MU = 200  # 集団内の個体数
    CXPB = 0.9  # 交叉率
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.average, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # 第一世代の生成
    pop1 = toolbox.population1(n=MU)
    pop1_init = pop1[:]
    invalid_ind1 = [ind for ind in pop1 if not ind.fitness.valid]

    pop2 = toolbox.population2(n=MU)
    pop2_init = pop2[:]
    invalid_ind2 = [ind for ind in pop2 if not ind.fitness.valid]
    fitnesses1 = toolbox.map(toolbox.evaluate1, invalid_ind1, len(invalid_ind1)*[best_out])
    for ind, fit in zip(invalid_ind1, fitnesses1):
        ind.fitness.values = fit
    pop1 = toolbox.select(pop1, len(pop1))

    fitnesses2 = toolbox.map(toolbox.evaluate2, len(invalid_ind2)*[pop1[0]],invalid_ind2)
    for ind, fit in zip(invalid_ind2, fitnesses2):
       ind.fitness.values = fit
    pop2 = toolbox.select(pop2, len(pop2))

    record = stats.compile(pop1)
    record = stats.compile(pop2)

    logbook.record(gen=0, evals=len(invalid_ind2), **record)
    print(logbook.stream)

    # 最適計算の実行
    for gen in range(1, NGEN):
        if gen % 2:
            # 子母集団生成
            offspring = tools.selTournamentDCD(pop1, len(pop1))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # 交叉と突然変異
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                # 交叉させる個体を選択
                if random.random() <= CXPB:
                    # 交叉
                    toolbox.mate(ind1, ind2)

                # 突然変異
                toolbox.mutates1(ind1)
                toolbox.mutates1(ind2)

                # 交叉と突然変異させた個体は適応度を削除する
                del ind1.fitness.values, ind2.fitness.values

            # 適応度を削除した個体について適応度の再評価を行う
            invalid_ind1 = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate1, invalid_ind1, len(invalid_ind1)*[best_out])
            for ind, fit in zip(invalid_ind1, fitnesses):
                ind.fitness.values = fit

            # 次世代を選択
            pop1 = toolbox.select(pop1 + offspring, MU)
            record = stats.compile(pop1)
            logbook.record(gen=gen, evals=len(invalid_ind1), **record)
            print(logbook.stream)
            best = tools.selBest(pop1, k=1)[0]
            print(f"Generation {gen}: Best individual: {best}")
        else:
            # 子母集団生成
            offspring = tools.selTournamentDCD(pop2, len(pop2))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # 交叉と突然変異
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                # 交叉させる個体を選択
                if random.random() <= CXPB:
                    # 交叉
                    toolbox.mate2(ind1, ind2)

                # 突然変異
                toolbox.mutates2(ind1)
                toolbox.mutates2(ind2)

                # 交叉と突然変異させた個体は適応度を削除する
                del ind1.fitness.values, ind2.fitness.values

            # 適応度を削除した個体について適応度の再評価を行う
            invalid_ind2 = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate2,  [best]*len(invalid_ind2), invalid_ind2)
            for ind, fit in zip(invalid_ind2, fitnesses):
                ind.fitness.values = fit

            # 次世代を選択
            pop2 = toolbox.select(pop2 + offspring, MU)
            record = stats.compile(pop2)
            logbook.record(gen=gen, evals=len(invalid_ind2), **record)
            print(logbook.stream)
            best_out = tools.selBest(pop2, k=1)[0]
            print(f"Generation {gen}: Best individual: {best_out}")

    print(pop1, pop1_init, stats)
    print(pop2, pop2_init, stats)

    fitnesses_init = np.array([list(pop1_init[i].fitness.values) for i in range(len(pop1_init))])
    fitnesses = np.array([list(pop1[i].fitness.values) for i in range(len(pop1))])
    plt.plot(fitnesses_init[:, 0], fitnesses_init[:, 1], "b.", label="Initial")
    plt.plot(fitnesses[:, 0], fitnesses[:, 1], "r.", label="Optimized")
    plt.legend(loc="upper right")
    plt.title("fitnesses")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.show()

    fitnesses_init = np.array([list(pop2_init[i].fitness.values) for i in range(len(pop2_init))])
    fitnesses = np.array([list(pop2[i].fitness.values) for i in range(len(pop2))])
    plt.plot(fitnesses_init[:, 0], fitnesses_init[:, 1], "b.", label="Initial")
    plt.plot(fitnesses[:, 0], fitnesses[:, 1], "r.", label="Optimized")
    plt.legend(loc="upper right")
    plt.title("fitnesses")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.show()

    # 最終世代のハイパーボリュームを出力
