import time
import random
from kesslergame import KesslerController, TrainerEnvironment
from kesslergame import Scenario, KesslerGame, GraphicsType
from FreshController import NewController

#隕石が2機体を囲むように，円状に並んで静止する
scenario1 = Scenario(name='Test Scenario',
                            asteroid_states=[
                                {'position':(400, 300), 'angle':180, 'speed':40},
                                {'position':(450, 250), 'angle':180, 'speed':40},
                                {'position': (500, 300), 'angle': 180, 'speed': 40},
                                {'position': (600, 400), 'angle': 180, 'speed': 40},
                                {'position': (500, 500), 'angle': 180, 'speed': 40},
                                {'position': (450, 550), 'angle': 180, 'speed': 40},
                                {'position': (400, 500), 'angle': 180, 'speed': 40},
                                {'position': (300, 400), 'angle': 180, 'speed': 40},

                            ],
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 5, 'team': 1},
                                {'position': (500, 400), 'angle': 90, 'lives':5, 'team': 2},
                            ],
                            map_size=(1000, 800),
                            #time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)
#隕石が縦に並び，横向きに流れる
scenario2 = Scenario(

        asteroid_states=[
            {'position':(100, 100), 'angle':180, 'speed':120},
            {'position':(100, 200), 'angle':180, 'speed':120},
            {'position': (100, 300), 'angle': 180, 'speed':120},
            {'position':(100, 400), 'angle':180, 'speed':120},
            {'position':(100, 500), 'angle':180, 'speed':120},
            {'position': (100, 600), 'angle': 180, 'speed':120},
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

#2機体を囲むような円が迫ってくる
scenario3 = Scenario(

        asteroid_states=[
            {'position': (895.9285948,456.9258097), 'angle':-171.81819999515864, 'speed': 20},
            {'position': (863.8528509,566.1658897), 'angle':-155.45456364305159, 'speed': 20},
            {'position': (802.2999129,661.9441976), 'angle':-139.09092728124793, 'speed': 20},
            {'position': (716.2564338,736.5013445), 'angle':-122.72729091386054, 'speed': 20},
            {'position': (612.6931445,783.7971537), 'angle':-106.36365454099587, 'speed': 20},
            {'position': (500.0001269,        800), 'angle':-90.00001817708605,  'speed': 20},
            {'position': (387.3070991,783.7972252), 'angle':-73.63638182411489,  'speed': 20},
            {'position': (283.7437798,736.5014818), 'angle':-57.272745457749366, 'speed': 20},
            {'position': (197.7002534,661.9443895), 'angle':-40.90910909178643,  'speed': 20},
            {'position': (136.1472546,566.1661207), 'angle':-24.54547273278905,  'speed': 20},
            {'position': (104.0714413,456.926061),  'angle':-8.181836370428783,  'speed': 20},
            {'position': (104.0714052,343.0741903), 'angle':8.181800004841364, 'speed': 20},
            {'position': (136.1471491,233.8341103), 'angle':24.545436356948443,'speed': 20},
            {'position': (197.7000871,138.0558024), 'angle':40.90907271875204, 'speed': 20},
            {'position': (283.7435662,63.49865549), 'angle':57.27270908691388, 'speed': 20},
            {'position': (387.3068555,16.20284632), 'angle':73.63634545819701, 'speed': 20},
            {'position': (499.9998731,2.09752E-11), 'angle':89.99998182291395, 'speed': 20},
            {'position': (612.6929009,16.20277479), 'angle':106.36361817548156,'speed': 20},
            {'position': (716.2562202,63.49851824), 'angle':122.72725454534827,'speed': 20},
            {'position': (802.2997466,138.0556105), 'angle':139.0908909082136, 'speed': 20},
            {'position': (863.8527454,233.8338793), 'angle':155.45452726721095,'speed': 20},
            {'position': (895.9285587,343.073939),  'angle':171.81816362957122,'speed': 20},



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
        settings = {
            "frequency": 60,
            "real_time_multiplier": 2,
            "graphics_on": True,
            "sound_on": True,
            "prints": True,
            "full_dashboard": True
        }

        # Instantiate an instance of FuzzyAsteroidGame
        game = KesslerGame()

        scenario_ship = scenario2

        run10_hit_env_child5 = [ 16.83714325,  34.80430902, 188.32179048,  86.97881193]
        run10_hit_env_child1 = [ 13.94107016,  53.73100038, 235.7052832,  101.09256617]
        run10_hit_best = [-6.73947456,  31.06759868, 170.07489844, 110.53856904]
        base = [0, 100, 200, 90]
        run_run_run_best = [ 11.04113667, 100.7399489,  180.64240626,  77.76295358]
        run_run_run_worst = [-18.71020633, 105.96800855, 204.54813249,  85.52341368]
        run_scenario2 = [-13.72084507,  82.02768699, 194.21582298,  67.33765658]
        run_scenario2_alpha1 = [ 11.0739878,   67.15342541, 143.03935401,  69.42809256]
        controllers = [NewController(run_scenario2), NewController(run_scenario2)]

        pre = time.perf_counter()
        score, perf_data = game.run(scenario=scenario_ship, controllers=controllers)
        print('Scenario eval time: '+str(time.perf_counter()-pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))

cil_run()
