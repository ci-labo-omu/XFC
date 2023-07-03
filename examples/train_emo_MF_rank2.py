import random
import numpy as np
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


    # this is a evaluate function
    def run_2out(gene, gene_out):
        accu, time = 0, 0
        for scene in Scenario2:
            f1, f2 = running(gene, gene_out, scene)
            accu += f1
            time += f2
        accu /= len(Scenario2)
        time /= len(Scenario2)

        return [accu, time]


    # これは制約関数で，制約を満たすかどうかを判定する

    NGEN = 500  # 繰り返し世代数
    MU = 100  # 集団内の個体数
    CXPB = 0.9  # 交叉率

    f1_values = np.zeros((MU, 2))
    f2_values = np.zeros((MU, 2))

    # make ranking from f1_values, f2_values, sorting each value
    # this is an individual to be evaluated
    # Individualは実数値のリストであり，適応度は2つの目的関数の値であり，目的関数は2つの評価値の，それぞれにおいての順位である　また，個体はf1, f2の値を属性として持つ
    # f1, f2は属性であり，それらを評価する関数を作る
    # それぞれの個体のf1,f2それぞれの順位が評価値となり，それの最小化問題を解く
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)


    # クラスIndividualに対し，属性f1, f2を追加して，それをメソッドで値を代入できるようにしてください

    class IndividualWithF(creator.Individual):
        f1 = 0.0
        f2 = 0.0

        def feasible(self):
            x1, x2, x3, x4, x5, x6 = self
            return x1 < x2 < x3 < x4 < x5 and -0.2 < x1 < 1.0 and -0.2 < x2 < 1.0 and -0.2 < x3 < 1.0 and -0.2 < x4 < 1.0 and -0.2 < x5 < 1.0 and 0 <= x6 <= 1.0

        # これは制約関数で，制約を満たす場合のみ評価関数を実行する
        def constrained_fitness(self):
            if self.feasible():
                self.f1, self.f2 = run_2out(self, best_out)
                return self.f1, self.f2
            else:
                return 0.0, 0.0

        def evaluate_individual_rank(self):
            f1, f2 = self.constrained_fitness()
            self.f1 = f1
            self.f2 = f2
            f1_rank = sum(f1 < f for f in f1_values)
            f2_rank = sum(f2 < f for f in f2_values)
            return f1_rank, f2_rank

        # 実際の評価関数は，f1, f2の値を，それぞれの目的関数の値として，それぞれの目的関数の値の順位を返すようにしてください
        # それぞれの目的関数の値の順位は，f_listの中の値と比較して，それぞれの目的関数の値が，f_listの中の値より小さいものの数を返すようにしてください
        # evaluate only rank, not play game.
        def evaluate_rank(self):
            f1_rank = sum(self.f1 < f for f in f1_values)
            f2_rank = sum(self.f2 < f for f in f2_values)
            return f1_rank, f2_rank




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

    toolbox.register("mutates", tools.mutPolynomialBounded, up=1, low=-0.2, indpb=1 / 6, eta=20)
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
    def plot_pf(f1_nondom, f2_nondom, gen):
        #第1パレートフロントのみをプロットする
        plt.plot(f1_nondom, f2_nondom, "r.",)
        plt.title(f"Fitness(NSGA2_Rank, GEN{gen})")
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
    for gen in range(1, NGEN+1):


        # 子母集団生成
        print(f"pop:{pop}")
        offspring = tools.selTournamentDCD(pop, len(pop))
        print(f"offsp1:{offspring}")

        offspring = [toolbox.clone(ind) for ind in offspring]
        print(f"offsp2:{offspring}")


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
        print(f"offsp3:{offspring}")
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
        print(f"{len(pop)}pop2{pop}")
        pop = toolbox.select(pop + offspring, MU)
        print(f"{len(pop)}pop3{pop}")

        print(f"Generation {gen}:")
        print(logbook.stream)
        non_dom = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
        print(non_dom)
        # 各世代ごとの個体の評価値を表示，f1,f2も表示する
        for j, ind in enumerate(non_dom[0]):
            print(f"Non-dominated individual {j + 1}: {ind} Rank: {ind.fitness.values} Fitness: {ind.f1} {ind.f2}")

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if(gen % 100 == 0):
            f1_nondom = np.array([ind.f1 for ind in non_dom[0]])
            f2_nondom = np.array([ind.f2 for ind in non_dom[0]])
            plot_pf(f1_nondom, f2_nondom,gen)

    print(pop, pop_init, stats)
    fitnesses_init = np.array([list(pop_init[i].fitness.values) for i in range(len(pop_init))])
    fitnesses = np.array([list(pop[i].fitness.values) for i in range(len(pop))])
    print(f"rankf1 = {f1_nondom}")
    print(f"rankf2 = {f2_nondom}")
    plot_pf(f1_nondom, f2_nondom, NGEN)