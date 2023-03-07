import time
import random
from kesslergame import KesslerController, TrainerEnvironment
from kesslergame import Scenario, KesslerGame, GraphicsType
from FreshController import NewController
from scenarios_official import *
from FreshController2 import NewController2

# 隕石が2機体を囲むように，円状に並んで静止する
scenario1 = Scenario(name='Test Scenario',
                     asteroid_states=[
                         {'position': (400, 300), 'angle': 180, 'speed': 40},
                         {'position': (450, 250), 'angle': 180, 'speed': 40},
                         {'position': (500, 300), 'angle': 180, 'speed': 40},
                         {'position': (600, 400), 'angle': 180, 'speed': 40},
                         {'position': (500, 500), 'angle': 180, 'speed': 40},
                         {'position': (450, 550), 'angle': 180, 'speed': 40},
                         {'position': (400, 500), 'angle': 180, 'speed': 40},
                         {'position': (300, 400), 'angle': 180, 'speed': 40},

                     ],
                     ship_states=[
                         {'position': (400, 400), 'angle': 90, 'lives': 5, 'team': 1},
                         {'position': (500, 400), 'angle': 90, 'lives': 5, 'team': 2},
                     ],
                     map_size=(1000, 800),
                     time_limit=60,
                     ammo_limit_multiplier=0,
                     stop_if_no_ammo=False)
# 隕石が縦に並び，横向きに流れる
scenario2 = Scenario(

    asteroid_states=[
        {'position': (100, 100), 'angle': 180, 'speed': 120},
        {'position': (100, 200), 'angle': 180, 'speed': 120},
        {'position': (100, 300), 'angle': 180, 'speed': 120},
        {'position': (100, 400), 'angle': 180, 'speed': 120},
        {'position': (100, 500), 'angle': 180, 'speed': 120},
        {'position': (100, 600), 'angle': 180, 'speed': 120},
        {'position': (100, 700), 'angle': 180, 'speed': 120},

    ],
    ship_states=[
        {'position': (400, 300), 'angle': 90, 'lives': 5, 'team': 1},
        {'position': (400, 500), 'angle': 90, 'lives': 5, 'team': 2},
    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False)

# 2機体を囲むような円が迫ってくる
scenario3 = Scenario(

    asteroid_states=[
        {'position': (895.9285948, 456.9258097), 'angle': -171.81819999515864, 'speed': 20},
        {'position': (863.8528509, 566.1658897), 'angle': -155.45456364305159, 'speed': 20},
        {'position': (802.2999129, 661.9441976), 'angle': -139.09092728124793, 'speed': 20},
        {'position': (716.2564338, 736.5013445), 'angle': -122.72729091386054, 'speed': 20},
        {'position': (612.6931445, 783.7971537), 'angle': -106.36365454099587, 'speed': 20},
        {'position': (500.0001269, 800), 'angle': -90.00001817708605, 'speed': 20},
        {'position': (387.3070991, 783.7972252), 'angle': -73.63638182411489, 'speed': 20},
        {'position': (283.7437798, 736.5014818), 'angle': -57.272745457749366, 'speed': 20},
        {'position': (197.7002534, 661.9443895), 'angle': -40.90910909178643, 'speed': 20},
        {'position': (136.1472546, 566.1661207), 'angle': -24.54547273278905, 'speed': 20},
        {'position': (104.0714413, 456.926061), 'angle': -8.181836370428783, 'speed': 20},
        {'position': (104.0714052, 343.0741903), 'angle': 8.181800004841364, 'speed': 20},
        {'position': (136.1471491, 233.8341103), 'angle': 24.545436356948443, 'speed': 20},
        {'position': (197.7000871, 138.0558024), 'angle': 40.90907271875204, 'speed': 20},
        {'position': (283.7435662, 63.49865549), 'angle': 57.27270908691388, 'speed': 20},
        {'position': (387.3068555, 16.20284632), 'angle': 73.63634545819701, 'speed': 20},
        {'position': (499.9998731, 2.09752E-11), 'angle': 89.99998182291395, 'speed': 20},
        {'position': (612.6929009, 16.20277479), 'angle': 106.36361817548156, 'speed': 20},
        {'position': (716.2562202, 63.49851824), 'angle': 122.72725454534827, 'speed': 20},
        {'position': (802.2997466, 138.0556105), 'angle': 139.0908909082136, 'speed': 20},
        {'position': (863.8527454, 233.8338793), 'angle': 155.45452726721095, 'speed': 20},
        {'position': (895.9285587, 343.073939), 'angle': 171.81816362957122, 'speed': 20},

    ],
    ship_states=[
        {'position': (300, 400), 'angle': 90, 'lives': 3, 'team': 1},
        {'position': (700, 400), 'angle': 90, 'lives': 3, 'team': 2},

    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False)


def timeout(input_data):
    # Function for testing the timing out by the simulation environment
    wait_time = random.uniform(0.02, 0.03)
    time.sleep(wait_time)


"""
class FuzzyController(ControllerBase):
    @property
    def name(self) -> str:
        return "Example Controller Name"

    def actions(self, ships: Tuple[SpaceShip], input_data: Dict[str, Tuple]) -> None:
        timeout(input_data)

        for ship in ships:
            ship.turn_rate = random.uniform(ship.turn_rate_range[0]/2.0, ship.turn_rate_range[1])
            ship.thrust = random.uniform(ship.thrust_range[0], ship.thrust_range[1])
            ship.fire_bullet = random.uniform(0.45, 1.0) < 0.5
"""


def cil_run():
    if __name__ == "__main__":
        # Available settings
        game_settings = {'perf_tracker': True,
                         'graphics_mode': GraphicsType.Tkinter,
                         'realtime_multiplier': 1}

        # Instantiate an instance of FuzzyAsteroidGame
        game = KesslerGame(settings=game_settings)

        scenario_ship = accuracy_test_3
        run10_hit_env_child5 = [16.83714325, 34.80430902, 188.32179048, 86.97881193]
        run10_hit_env_child1 = [13.94107016, 53.73100038, 235.7052832, 101.09256617]
        run10_hit_best = [-6.73947456, 31.06759868, 170.07489844, 110.53856904]
        base = [0, 100, 200, 90]
        train_both_best = [14.0575167, 126.53980694, 173.8329372, 85.8197373]
        train_both_out = [-471.40500295, -393.36724752, 80.78269604, -394.46713071, -94.65366519,
                          145.60176213, 104.763696, 84.08122696, -34.68533779, 86.22654228,
                          145.54315815, 150.49783484, 120.49715349, 134.90680705, 125.91555583,
                          121.16169725, 132.32261205, 127.56778426]
        run_scenario2 = [-13.72084507, 82.02768699, 194.21582298, 67.33765658]
        run_scenario2_alpha1 = [11.0739878, 67.15342541, 143.03935401, 69.42809256]
        run_scenario3 = [-69.35648117, 78.38392824, 243.45673183, 35.80583847]
        gene_standard = [-480, -360, 120, -360, -60, 180, 120, 120, 0, 90, 180, 180, 180, 180, 180, 180, 180, 180]
        out_run_scenario3 = [-476.26230498, -374.16227015, 91.14998438, -339.49869418, -65.13170907,
                             188.37775874, 102.19974504, 111.27493103, -30.16147097, 85.87972781,
                             139.16449379, 147.9620688, 160.80626404, 161.90306892, 167.57717483,
                             149.0889584, 172.93617503, 177.46649927, ]
        official = [16.95719695, 37.1543788, 172.7201554, 105.07226885]
        gene = [-6.73947456, 31.06759868, 170.07489844, 110.53856904, 20, 60, 100]
        out_gene27 = [-475.08694037, -442.53759351, -396.0480849, -393.34234207, -341.39336395,
                      -320.58626158, -262.68262421, -234.57091884, -201.1131611, -162.76750302,
                      -141.35728289, -131.68628541, -100.00442, -18.39533764, 4.18639545,
                      22.67664509, 60.01319639, 101.30350708, 137.29556858, 139.27864197,
                      208.54313806, 230.1011409, 258.17616278, 291.08498313, 323.81475829,
                      367.86732364, 405.76717874, -12.6830166, 1.83333114, 5.78869101,
                      5.24857734, 22.30789036, 31.1433117, 29.60538188, 49.15401268,
                      41.02889741, 44.69831398, 60.69954653, 52.47406597, 84.61665249,
                      70.1936873, 77.04816783, 84.38645706, 98.72519946, 106.6072971,
                      107.71388976, 119.67331814, 125.89372316, 132.42602437, 133.90306922,
                      151.46181583, 158.17796556, 162.05969333, 170.08653585]
        out_gomi_27 = [-450.96207783, -438.93322999, -410.98380434, -359.34291906, -342.70869446,
                       -289.52428687, -258.35338152, -247.69027587, -200.2164681, -152.75393967,
                       -125.40591373, -111.5491068, -54.48714418, -30.41551278, -7.57175439,
                       59.6561563, 82.48503769, 91.45233788, 144.22871106, 167.89977464,
                       178.43038745, 237.74748956, 268.07203552, 300.44230412, 317.19801648,
                       321.36336404, 395.56240468, -14.83924087, -7.15153114, -13.28413844,
                       9.2237864, 16.88265306, 17.24603978, 25.75266376, 31.15429665,
                       41.11550162, 36.41059053, 46.7358634, 63.75968181, 69.54666296,
                       78.61622724, 96.94769441, 95.55243202, 95.84630305, 98.61177394,
                       114.87964996, 113.62692123, 121.30212485, 127.82396108, 173.29035104,
                       125.53045348, 143.69142817, 159.29737721, 164.53682925]

        controllers = [NewController2(gene, out_gene27), NewController2(gene, out_gene27)]

        pre = time.perf_counter()
        score, perf_data = game.run(scenario=scenario_ship, controllers=controllers)
        print('Scenario eval time: ' + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))


cil_run()
