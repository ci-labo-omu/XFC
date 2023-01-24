import math
import random
#from deap import base, creator
from Controller import NewController
from src.kesslergame import TrainerEnvironment
from src.kesslergame import Scenario
import time
import numpy as np
import csv

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

    def running(gene):
        #creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        #creator.create("Individual", list, fitness=creator.FitnessMax)
        game = TrainerEnvironment(settings=settings)
        scenario = Scenario(name="Multi-Ship",
                 time_limit = 15,
                 num_asteroids=5,
                 ship_states=[{"position": (300, 400), "angle": 0, "lives": 3, "team": 1},
                              {"position": (700, 400), "angle": 0, "lives": 3, "team": 2},
                              ],
                 ammo_limit_multiplier=1.0)
        # Because of how the arcade library is implemented, there are memory leaks for instantiating the environment
        # too many times, keep instantiations to a small number and simply reuse the environment
        controllers = [NewController(gene), NewController(gene)]
        score, perf_data = game.run(controllers=controllers, scenario=scenario)

        """print('Scenario eval time: ' + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))"""

        return sum([team.asteroids_hit for team in score.teams])


    run10_hit_best = [-6.73947456,  31.06759868, 170.07489844, 110.53856904]
    run10_hit_ave = [36.32153689,  93.61669382, 126.92704083,  83.56178968]
    run10_hit_worst = [-41.04162795,  48.13773711, 146.60937725,  90.87338001]



    #simple1
    #gene = [7.76711477e-02, 3.92488857e+01, 2.26744742e+02, 1.03120448e+02]
    #simple5
    #gene = [-17.21466721,  61.88116424, 174.37963555,  94.87146695]
    #simple100env_1
    #gene = [ -6.03595681,  59.09804168, 179.83703166,  88.50611241]
    #simple100env_5
    #gene = [-24.42916563,  62.61450636, 180.93782134,  81.10558]

    scores = [0.0 for i in range(3000)]
    sum1 = 0.0
    for i in range(3000):
        rand = np.random.randint(5, 10)
        score = running(run10_hit_worst)
        print(i)
        print(score)
        scores[i] = score
    ave = np.average(scores)
    std = np.std(scores)
    maximum = np.max(score)
    minimum = np.min(score)
    print("平均:", ave)
    print("標準偏差:", std)
    print("最高:", maximum)
    print("最低:", minimum)




#iwakutukinosaiakusentaku
#平均: 0.5115938999572495  標準偏差: 0.10157157874592687

#run100_child1
#平均: 0.5024605328406049  標準偏差: 0.09698928079134518



#simple_child5
#平均: 0.5052262332554018 標準偏差: 0.10739871102384166

#simple_child1_2回目
#平均: 0.5032122454361122 標準偏差: 0.24541140260070002