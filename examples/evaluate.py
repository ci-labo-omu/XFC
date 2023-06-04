import math
import random
from FreshController import NewController
from FreshController3 import NewController3
from src.kesslergame import TrainerEnvironment, KesslerController
from src.kesslergame import Scenario
from Scenarios import *
from Scenarios_ import *
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
    genes2 = [-0.7913319243922498, -0.07193139833080146, -0.657673503108559, -0.9106218912443529, 0.4947505711029901, -0.01323508087867964, 0.9966524525367539, 0.8571038910937839, -0.31138363993272267, 0.3293051529245099, 0.6196613061381991, -0.6618518784482988, 0.8308038571992254, 0.9335824222495888, -0.120772705208406, -0.8084364402945512, -0.5998728982212574, -0.4083684566859418, 0.9911016676992435, -0.8028631776645034, -0.2222819914181749, 0.9973080570151198, 0.6796069081195464, -0.45808777189198463, -0.5866563187391443, -0.33263267981343403, -0.13032082476894624, 0.012335279256734721, -0.6795485184921826, 0.13487803869939163]
    gene = [-0.1042433486127105, 0.0008622096524024433, 0.10294497774552946, 0.2041584191609273, 0.6202688983342248, 0.8216965488701551]
    new_gene = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]


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




    def evaluate(gene, Scenarioset):
        scores = [run_2outs(gene, Scenarios) for Scenarios in Scenarioset]
        return scores



    def run_2outs(gene, Scenarios):
        F1, F2 = [],[]
        for scene in Scenarios:
            f1, f2 = running(gene, new_gene, scene)
            F1.append(f1)
            F2.append(f2)
        return F1,F2
    #NSGA-ii_f1Best
    gene_f1 = [-0.14526328149051398, 0.03340750806686922, 0.04022883828200606, 0.21691211489553586, 0.99867173107487, 0.36928784240304047]
    #NSGA-ii_Rank_f2Best
    gene_f2 = [-0.07235148723096443, 0.04702269777018892, 0.09718268109285483, 0.5986051177058564, 0.9791829632646416, 0.7418960616060596]
    #NSGA-ii_Rank f1Best
    gene_rank_f1 = [-0.06798849258168843, 0.0310711187313323, 0.07348882223802997, 0.6727910463116129, 0.9679661221078124, 0.3025361027523371]
    #NSGA-ii f2Best
    gene_rank_f2 = [-0.1978109424696998, 0.03802359461653661, 0.1124835402899757, 0.41296220873308775, 0.9585202424577131, 0.3849678700149811]

    gene_S1_f1 = [0.12684129112406245, 0.18135808528642122, 0.570530299296549, 0.5866575601204076, 0.9466493644093196,
                  0.4097363092324291]
    gene_S1_f2 = [0.06497155387134385, 0.18194149992421801, 0.5175972437409366, 0.6531517111503555, 0.8718438902781543,
                  0.4081284266603572]
    gene_S1_rank_f1 = [-0.16498214926485233, 0.12312671686440918, 0.1649348982529897, 0.6947317875931055,
                       0.9949955040578345, 0.1894984559340283]
    gene_S1_rank_f2 = [-0.1643965145721291, 0.1514811297593448, 0.22904887038919455, 0.47981093624301113,
                       0.7739551979819967, 0.3445692868071561]

    gene_S2_f1 = [-0.13505129718670034, 0.04703967306525229, 0.048838982142104415, 0.48871649637432907,
                  0.9962437082247129, 0.2506812574844112]
    gene_S2_f2 = [-0.02983863306835434, 0.013128007497429775, 0.10510869371455275, 0.49418222816205043,
                  0.8164106285038217, 0.33372092991141833]
    gene_S2_rank_f1 = [-0.011974941455293778, 0.04995826281654835, 0.6678567396503903, 0.7526574937409152,
                       0.854539770061158, 0.477250061489301]
    gene_S2_rank_f2 = [0.023907791854285904, 0.1044149695479607, 0.16138462457676261, 0.5069466146556928,
                       0.9805029578955049, 0.45085062178110785]

    gene_S3_f1 = [0.016032300651053186, 0.02350906627703432, 0.09228189142541968, 0.6146116652210155, 0.774641810371584,
                  0.2807384628947408]
    gene_S3_f2 = [-0.19769411880568696, 0.04478636386739436, 0.0965065259938356, 0.4345352269784616, 0.8730396252055118,
                  0.36602003734038413]
    gene_S3_rank_f1 = [-0.06778689764304852, 0.06867619128391636, 0.08567125041979982, 0.28377889238834414,
                       0.41960537522950475, 0.9945664067809892]
    gene_S3_rank_f2 = [-0.18468106821840738, 0.0690375415616099, 0.08035678279244657, 0.8279724745391738,
                       0.8667879484471496, 0.6983771730515432]
    gene_list = [gene_f1, gene_f2, gene_rank_f1, gene_rank_f2,]

    i = 1
    for gene in gene_list:
        print(i)
        scores = run_2outs(gene, Scenario_accuracy)
        print(scores)
        i += 1