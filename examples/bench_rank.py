import random
import numpy as np
from deap.benchmarks import zdt3, zdt4, zdt2, zdt1, zdt6, dtlz1
#ZDT1:30, ZDT2:30, ZDT3:30, ZDT4:10, ZDT6:10

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

    NGEN = 1001  # 繰り返し世代数
    MU = 100  # 集団内の個体数
    CXPB = 0.9  # 交叉率



    # make ranking from f1_values, f2_values, sorting each value
    # this is an individual to be evaluated
    # Individualは実数値のリストであり，適応度は2つの目的関数の値であり，目的関数は2つの評価値の，それぞれにおいての順位である　また，個体はf1, f2の値を属性として持つ
    # f1, f2は属性であり，それらを評価する関数を作る
    # それぞれの個体のf1,f2それぞれの順位が評価値となり，それの最小化問題を解く


    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # クラスIndividualに対し，属性f1, f2を追加して，それをメソッドで値を代入できるようにする
    class IndividualWithF(creator.Individual):
        f1 = 0.0
        f2 = 0.0

        # 実際の評価関数は，f1, f2の値を，それぞれの目的関数の値として，それぞれの目的関数の値の順位を返すようにしてください
        # それぞれの目的関数の値の順位は，f_listの中の値と比較して，それぞれの目的関数の値が，f_listの中の値より小さいものの数を返すようにしてください
        def evaluate_rank(self):
            f1, f2 = dtlz1(self, 2)
            self.f1 = f1
            self.f2 = f2
            f1_rank = sum(self.f1 > f for f in f1_values)
            f2_rank = sum(self.f2 > f for f in f2_values)
            return f1_rank, f2_rank




    toolbox = base.Toolbox()
    #ZDT4を解くためのインスタンスを作る

    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, IndividualWithF, toolbox.attr_float, n=30)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", IndividualWithF.evaluate_rank)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)

    toolbox.register("mutates", tools.mutPolynomialBounded, low=0, up=1, indpb=1.0 / 10.0, eta=20)
    toolbox.register("select", tools.selNSGA2, )

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
    def plot_pf():
        f1_values, f2_values = values_pop(pop)
        plt.plot(f1_values, f2_values, "r.")

        plt.title(f"Fitnesses(Generation:{gen})")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.grid(True)
        plt.show()


    # 第一世代の生成
    pop = toolbox.population(n=MU)
    pop_init = pop[:]
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    #はじめのf_Valuesはダミー
    f1_values, f2_values = values_pop(pop)

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)

    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)


    # 最適計算の実行
    for gen in range(1, NGEN):
        if(gen == 100 or gen == 200 or gen == 500):
            plot_pf()

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
        # ここでf_listを更新する
        f1_values, f2_values = values_pop(pop)

           # 母集団の全要素のfitness.valuesを，f_listを使って更新する
        for ind in pop:
            ind.fitness.values = ind.evaluate_rank()

        # 次世代を選択
        pop = toolbox.select(pop + offspring, MU)
        print(f"Generation {gen}:")
        print(logbook.stream)
        non_dom = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
        # 各世代ごとの個体の評価値を表示，f1,f2も表示する
        for j, ind in enumerate(non_dom[0]):
            print(f"Non-dominated individual {j + 1}: {ind} Rank: {ind.fitness.values} Fitness: {ind.f1} {ind.f2}")

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

    print(tools.ParetoFront().items)

    # 最終的なf1, f2のパレートフロントをプロットする
    plot_pf()

