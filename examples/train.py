import math
import random
from FreshController import NewController
from src.kesslergame import TrainerEnvironment, KesslerController
from src.kesslergame import Scenario
from Scenarios import *
from scenarios_official import *
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
    out_base = [-450, -360, 120, -360, -60, 180, 120, 120, 0, 90, 160, 160, 140, 140, 140, 140, 140, 140]


    def running(gene, scenario):
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Individual", list, fitness=creator.FitnessMax)
        game = TrainerEnvironment()
        # Because of how the arcade library is implemented, there are memory leaks for instantiating the environment
        # too many times, keep instantiations to a small number and simply reuse the environment
        controllers = [NewController(gene, out_base), NewController(gene, out_base)]
        score, perf_data = game.run(scenario=scenario, controllers=controllers)

        """print('Scenario eval time: ' + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))"""

        return sum([team.asteroids_hit for team in score.teams])


    def run_collection(gene):
        score = [running(gene, scene) for scene in Scenario_list]
        return np.average(score)


    def run10worst(gene):
        min = 240
        for i in range(10):
            score = running(gene)
            if min > score: min = score
        return min


    def run10best(gene):
        max = 0
        for i in range(10):
            score = running(gene)
            if max < score: max = score
        return max


    def blend(x_a, x_b, alpha):
        u = []
        for i in range(4):
            A = np.minimum(x_a[i], x_b[i]) - alpha * np.fabs(x_a[i] - x_b[i])
            B = np.maximum(x_a[i], x_b[i]) + alpha * np.fabs(x_a[i] - x_b[i])
            u.append(np.random.uniform(A, B))
        # u = np.clip(u, x_min, x_max)
        return u


    def tournament(_genes, _accu):
        a, b, d, e, f, g = rng.choice(len(_genes), 6, replace=False)
        if _accu[d] > _accu[e]:
            gene_a = genes[d]
        else:
            gene_a = genes[e]
        if _accu[f] > _accu[g]:
            gene_b = genes[f]
        else:
            gene_b = genes[g]
        return gene_a, gene_b


    standard = np.array([0, 100, 200, 90])
    genes = np.tile(standard, (10, 1))
    genes = (2 * np.random.rand(10, 4) - 1) * np.array([30, 30, 30, 10]) + genes
    ave_list = []
    print(run_collection(genes[0]))
    accu = [run_collection(genes[j]) for j in range(len(genes))]

    for i in range(2500):

        gene_a, gene_b = tournament(genes, accu)
        children = blend(gene_a, gene_b, 0.5)
        if children[0] > children[1] or children[1] > children[2]:
            i -= 1
            continue
        score_child = run_collection(children)
        if score_child > np.min(accu):
            genes[np.argmin(accu)] = children
            accu[np.argmin(accu)] = score_child

        first = np.argmax(accu)
        print(i)
        print(first)
        print(genes[first])

        with open("Officials.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accu)
