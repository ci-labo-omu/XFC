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
from FreshController5 import NewController5

# メンバシップ関数と出力を両方進化させる
# メンバシップ関数変更ｰ>評価ｰ>出力ｰ>評価で1世代
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
        controllers = [NewController5(gene, genes2), NewController5(gene, genes2)]
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
        f2 = 1 - score.sim_time / 120 if sum([team.deaths for team in score.teams]) < 3 else  score.sim_time / 120
        return f1, f2

#あとは，run_2outから2つのあれを出すだけ
    new_gene = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]
    #いわくつきのemo4
    gene_emo = [131.38690516561581, 37.305892544275906, -34.71117313825014, -128.4534722929572, 311.07592204954716, 374.2384352762294]
    def run_2out(gene_out):
        accu, time = 0, 0
        for scene in Scenario_list:
            f1, f2 = running(gene_emo, gene_out, scene)
            accu += f1
            time += f2
        accu /= len(Scenario_list)
        time /= len(Scenario_list)
        return accu, time


    def feasible(individual):
        gene = np.array(individual)
        return abs(gene[0:15]).all() < 480 and abs(gene[16:30]).all() < 180

    # 制約適応度
    def constrained_fitness(individual):
        if not feasible(individual):
            return 0, 0
        else:
            return run_2out(individual)

    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    MIN, MAX = -1.0, 1.0
    dim = 6
    toolbox.register("attr_float", random.uniform, MIN, MAX)
    toolbox.register("attr_angle", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (15*[toolbox.attr_float]+15*[toolbox.attr_angle]), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", run_2out)
    toolbox.register("evaluate", constrained_fitness)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=MIN, up=MAX, eta=20.0)
    toolbox.register("mutates", tools.mutPolynomialBounded, up=MAX, low=MIN, indpb=1 / 15, eta= 30)
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


    new_gene = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]

    standard = np.array([0, 100, 200, 350, 500, 90])

    NGEN = 200  # 繰り返し世代数
    MU = 100  # 集団内の個体数
    CXPB = 0.9  # 交叉率
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.average, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

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
            del ind1.fitness.values, ind2.fitness.values

        # 適応度を削除した個体について適応度の再評価を行う
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 次世代を選択
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        best_individual = tools.selBest(pop, k=1)[0]
        print(f"Generation {gen}: Best individual: {best_individual}, Best gene: {max(best_individual)}")

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
    # 最終世代のハイパーボリュームを出力


