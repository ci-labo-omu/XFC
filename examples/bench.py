import random
import numpy as np
from deap.benchmarks import zdt3, zdt4, zdt2, zdt6, dtlz1

from Scenarios import *
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt
from FreshController5 import NewController5
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








    # これは制約関数で，制約を満たすかどうかを判定する

    NGEN = 500  # 繰り返し世代数
    MU = 100  # 集団内の個体数
    CXPB = 0.9  # 交叉率



    # make ranking from f1_values, f2_values, sorting each value
    # this is an individual to be evaluated
    # Individualは実数値のリストであり，適応度は2つの目的関数の値であり，目的関数は2つの評価値の，それぞれにおいての順位である　また，個体はf1, f2の値を属性として持つ
    # f1, f2は属性であり，それらを評価する関数を作る
    # それぞれの個体のf1,f2それぞれの順位が評価値となり，それの最小化問題を解く


    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)




    toolbox = base.Toolbox()
    #ZDT4を解くためのインスタンスを作る

    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=30)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", zdt2,)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)

    toolbox.register("mutates", tools.mutPolynomialBounded, up=1, low=0, indpb=1.0 / 10.0, eta=20)
    toolbox.register("select", tools.selNSGA2, )

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
            print(ind1, ind2)
            if random.random() <= CXPB:
                # 交叉
                toolbox.mate(ind1, ind2)
            # 突然変異
            toolbox.mutates(ind1)
            toolbox.mutates(ind2)
            print(ind1, ind2)

            # 交叉と突然変異させた個体は適応度を削除する
            del ind1.fitness.values, ind2.fitness.values
        # 適応度を削除した個体について適応度の再評価を行う
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # 次世代を選択
        pop = toolbox.select(pop + offspring, MU)
        print(f"Generation {gen}:")
        print(logbook.stream)
        non_dom = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
        for j, ind in enumerate(non_dom[0]):
            print(f"Non-dominated individual {j + 1}: {ind} Fitness:      {np.array(ind.fitness.values)}")

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)


    # 最終的なf1, f2のパレートフロントをプロットする
    print(f"pop = {pop}")
    fitnesses_init = np.array([list(pop_init[i].fitness.values) for i in range(len(pop_init))])
    fitnesses = np.array([list(pop[i].fitness.values) for i in range(len(pop))])

    plt.plot(fitnesses[:, 0], fitnesses[:, 1], "r.")
    plt.title("Fitnesses")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.show()
