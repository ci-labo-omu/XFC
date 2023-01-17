import math
from typing import Tuple, Dict
import random
from kesslergame import KesslerController
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from func import angle360
from kesslergame import Ship


class NewController(KesslerController):
    """
    Class to be used by UC Fuzzy Challenge competitors to create a fuzzy logic controller
    for the Asteroid Smasher game.

    Note: Your fuzzy controller class can be called anything, but must inherit from
    the ``ControllerBase`` class (imported above)

    Users must define the following:
    1. __init__()
    2. actions(self, ship: SpaceShip, input_data: Dict[str, Tuple])

    By defining these interfaces, this class will work correctly
    """

    """
         A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
        game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
        game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
        """

    def __init__(self):
        """
        Create your fuzzy logic controllers and other objects here
        """

        # Closeness to where evasive behavior is triggered. Variable by size. Default: 20
        def leftrightcheck(shipangle, astangle):
            diff = shipangle - astangle + self.ant_angle
            absdiff = abs(diff)
            if absdiff < 180 and diff > 0:
                leftright = 1
            elif absdiff < 180 and diff < 0:
                leftright = 0
            elif absdiff > 180 and diff > 0:
                leftright = 0
            elif absdiff > 180 and diff < 0:
                leftright = 1
            else:
                leftright = 0
                print('problem in left right checker')
            return leftright

        self.leftright = leftrightcheck

        # テストとしてTeam Asimovから引用
        names = ['near', 'close', 'far']
        A1 = ctrl.Antecedent(np.arange(0, 151, 1), 'A1')

        A1.automf(names=names)
        A1.view()

        B1 = ctrl.Antecedent(np.arange(-91, 271, 1), 'B1')
        B1['front'] = fuzz.trimf(B1.universe, [-15, 0, 15])
        B1['middle'] = fuzz.trimf(B1.universe, [15, 45, 75])
        B1['back'] = fuzz.trimf(B1.universe, [75, 180, 180])

        Con1 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con1')
        Con1['Very Low'] = fuzz.trimf(Con1.universe, [-25, 0, 25])
        Con1['Low'] = fuzz.trimf(Con1.universe, [0, 25, 50])
        Con1['Medium'] = fuzz.trimf(Con1.universe, [25, 50, 75])
        Con1['High'] = fuzz.trimf(Con1.universe, [50, 75, 100])
        Con1['Very High'] = fuzz.trimf(Con1.universe, [75, 100, 125])

        Con1_rule1 = ctrl.Rule(antecedent=(A1['near'] & B1['front']),
                               consequent=Con1['Very High'], label='Con1_rule1')
        Con1_rule2 = ctrl.Rule(antecedent=(A1['close'] & B1['front']),
                               consequent=Con1['High'], label='Con1_rule2')
        Con1_rule3 = ctrl.Rule(antecedent=(A1['far'] & B1['front']),
                               consequent=Con1['Medium'], label='Con1_rule3')
        Con1_rule4 = ctrl.Rule(antecedent=(A1['near'] & B1['middle']),
                               consequent=Con1['Very High'], label='Con1_rule4')
        Con1_rule5 = ctrl.Rule(antecedent=(A1['close'] & B1['middle']),
                               consequent=Con1['High'], label='Con1_rule5')
        Con1_rule6 = ctrl.Rule(antecedent=(A1['far'] & B1['middle']),
                               consequent=Con1['Medium'], label='Con1_rule6')
        Con1_rule7 = ctrl.Rule(antecedent=(A1['near'] & B1['back']),
                               consequent=Con1['Very High'], label='Con1_rule7')
        Con1_rule8 = ctrl.Rule(antecedent=(A1['close'] & B1['back']),
                               consequent=Con1['High'], label='Con1_rule8')
        Con1_rule9 = ctrl.Rule(antecedent=(A1['far'] & B1['back']),
                               consequent=Con1['Medium'], label='Con1_rule9')

        Con1_system = ctrl.ControlSystem(rules=[Con1_rule1, Con1_rule2, Con1_rule3, Con1_rule4,
                                                Con1_rule5, Con1_rule6, Con1_rule7, Con1_rule8,
                                                Con1_rule9])
        self.Con1_sim = ctrl.ControlSystemSimulation(Con1_system)

        A2 = ctrl.Antecedent(np.arange(0, 361), 'A2')
        A2.automf(names=names)
        # shortest_distance < 50 + (12 * clast_size)
        B2 = ctrl.Antecedent(np.arange(-91, 271, 1), 'B2')
        B2['front'] = fuzz.trimf(B1.universe, [-15, 0, 15])
        B2['middle'] = fuzz.trimf(B1.universe, [15, 45, 75])
        B2['back'] = fuzz.trimf(B1.universe, [75, 180, 285])

        Con2 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con2')
        Con2['Very Low'] = fuzz.trimf(Con2.universe, [-25, 0, 25])
        Con2['Low'] = fuzz.trimf(Con2.universe, [0, 25, 50])
        Con2['Medium'] = fuzz.trimf(Con2.universe, [25, 50, 75])
        Con2['High'] = fuzz.trimf(Con2.universe, [50, 75, 100])
        Con2['Very High'] = fuzz.trimf(Con2.universe, [75, 100, 125])
        Con2_rule1 = ctrl.Rule(antecedent=(A2['near'] & B2['front']),
                               consequent=Con2['Very High'], label='Con2_rule1')
        Con2_rule2 = ctrl.Rule(antecedent=(A2['close'] & B2['front']),
                               consequent=Con2['High'], label='Con2_rule2')
        Con2_rule3 = ctrl.Rule(antecedent=(A2['far'] & B2['front']),
                               consequent=Con2['Medium'], label='Con2_rule3')
        Con2_rule4 = ctrl.Rule(antecedent=(A2['near'] & B2['middle']),
                               consequent=Con2['Very High'], label='Con2_rule4')
        Con2_rule5 = ctrl.Rule(antecedent=(A2['close'] & B2['middle']),
                               consequent=Con2['High'], label='Con2_rule5')
        Con2_rule6 = ctrl.Rule(antecedent=(A2['far'] & B2['middle']),
                               consequent=Con2['Medium'], label='Con2_rule6')
        Con2_rule7 = ctrl.Rule(antecedent=(A2['near'] & B2['back']),
                               consequent=Con2['Very High'], label='Con2_rule7')
        Con2_rule8 = ctrl.Rule(antecedent=(A2['close'] & B2['back']),
                               consequent=Con2['High'], label='Con2_rule8')
        Con2_rule9 = ctrl.Rule(antecedent=(A2['far'] & B2['back']),
                               consequent=Con2['Medium'], label='Con2_rule9')
        Con2_system = ctrl.ControlSystem(rules=[Con2_rule1, Con2_rule2, Con2_rule3, Con2_rule4,
                                                Con2_rule5, Con2_rule6, Con2_rule7, Con2_rule8,
                                                Con2_rule9])
        self.Con2_sim = ctrl.ControlSystemSimulation(Con2_system)

        B3 = ctrl.Antecedent(np.arange(-91, 271, 1), 'B3')
        B3['front'] = fuzz.trimf(B3.universe, [-15, 0, 15])
        B3['middle'] = fuzz.trimf(B3.universe, [15, 45, 75])
        B3['back'] = fuzz.trimf(B3.universe, [75, 180, 285])
        A3 = ctrl.Antecedent(np.arange(0, 261, 1), 'A3')
        A3.automf(names=names)

        Con3 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con3')
        Con3['Very Low'] = fuzz.trimf(Con3.universe, [-25, 0, 25])
        Con3['Low'] = fuzz.trimf(Con3.universe, [0, 25, 50])
        Con3['Medium'] = fuzz.trimf(Con3.universe, [25, 50, 75])
        Con3['High'] = fuzz.trimf(Con3.universe, [50, 75, 100])
        Con3['Very High'] = fuzz.trimf(Con3.universe, [75, 100, 125])

        Con3_rule1 = ctrl.Rule(antecedent=(A3['near'] & B3['front']),
                               consequent=Con3['Very High'], label='Con3_rule1')

        Con3_rule2 = ctrl.Rule(antecedent=(A3['close'] & B3['front']),
                               consequent=Con3['High'], label='Con3_rule2')

        Con3_rule3 = ctrl.Rule(antecedent=(A3['far'] & B3['front']),
                               consequent=Con3['Medium'], label='Con3_rule3')

        Con3_rule4 = ctrl.Rule(antecedent=(A3['near'] & B3['middle']),
                               consequent=Con3['Very High'], label='Con3_rule4')

        Con3_rule5 = ctrl.Rule(antecedent=(A3['close'] & B3['middle']),
                               consequent=Con3['High'], label='Con3_rule5')

        Con3_rule6 = ctrl.Rule(antecedent=(A3['far'] & B3['middle']),
                               consequent=Con3['Medium'], label='Con3_rule6')

        Con3_rule7 = ctrl.Rule(antecedent=(A3['near'] & B3['back']),
                               consequent=Con3['Very High'], label='Con3_rule7')

        Con3_rule8 = ctrl.Rule(antecedent=(A3['close'] & B3['back']),
                               consequent=Con3['High'], label='Con3_rule8')

        Con3_rule9 = ctrl.Rule(antecedent=(A3['far'] & B3['back']),
                               consequent=Con3['Medium'], label='Con3_rule9')

        Con3_system = ctrl.ControlSystem(rules=[Con3_rule1, Con3_rule2, Con3_rule3, Con3_rule4,
                                                Con3_rule5, Con3_rule6, Con3_rule7, Con3_rule8,
                                                Con3_rule9])
        self.Con3_sim = ctrl.ControlSystemSimulation(Con3_system)

        B4 = ctrl.Antecedent(np.arange(-91, 271, 1), 'B4')
        B4['front'] = fuzz.trimf(B4.universe, [-15, 0, 15])
        B4['middle'] = fuzz.trimf(B4.universe, [15, 45, 75])
        B4['back'] = fuzz.trimf(B4.universe, [75, 180, 285])

        A4 = ctrl.Antecedent(np.arange(0, 361, 1), 'A4')
        A4.automf(names=names)

        Con4 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con4')
        Con4['Very Low'] = fuzz.trimf(Con4.universe, [-25, 0, 25])
        Con4['Low'] = fuzz.trimf(Con4.universe, [0, 25, 50])
        Con4['Medium'] = fuzz.trimf(Con4.universe, [25, 50, 75])
        Con4['High'] = fuzz.trimf(Con4.universe, [50, 75, 100])
        Con4['Very High'] = fuzz.trimf(Con4.universe, [75, 100, 125])

        Con4_rule1 = ctrl.Rule(antecedent=(A4['near'] & B4['front']),
                               consequent=Con4['Very High'], label='Con4_rule1')

        Con4_rule2 = ctrl.Rule(antecedent=(A4['close'] & B4['front']),
                               consequent=Con4['High'], label='Con4_rule2')

        Con4_rule3 = ctrl.Rule(antecedent=(A4['far'] & B4['front']),
                               consequent=Con4['Very Low'], label='Con4_rule3')

        Con4_rule4 = ctrl.Rule(antecedent=(A4['near'] & B4['middle']),
                               consequent=Con4['High'], label='Con4_rule4')

        Con4_rule5 = ctrl.Rule(antecedent=(A4['close'] & B4['middle']),
                               consequent=Con4['Medium'], label='Con4_rule5')

        Con4_rule6 = ctrl.Rule(antecedent=(A4['far'] & B4['middle']),
                               consequent=Con4['Very Low'], label='Con4_rule6')

        Con4_rule7 = ctrl.Rule(antecedent=(A4['near'] & B4['back']),
                               consequent=Con4['High'], label='Con4_rule7')

        Con4_rule8 = ctrl.Rule(antecedent=(A4['close'] & B4['back']),
                               consequent=Con4['Low'], label='Con4_rule8')

        Con4_rule9 = ctrl.Rule(antecedent=(A4['far'] & B4['back']),
                               consequent=Con4['Very Low'], label='Con4_rule9')

        Con4_system = ctrl.ControlSystem(rules=[Con4_rule1, Con4_rule2, Con4_rule3, Con4_rule4,
                                                Con4_rule5, Con4_rule6, Con4_rule7, Con4_rule8,
                                                Con4_rule9])
        self.Con4_sim = ctrl.ControlSystemSimulation(Con4_system)

    def actions(self, ownship: Ship, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        ship_list = input_data["ships"]
        ast_list = input_data["asteroids"]
        dist_list = np.array([math.dist(ownship['position'], ast['position']) for ast in ast_list])
        closest = np.argmin(dist_list)
        dist_closest = dist_list[closest]
        ang = np.array(ast_list[closest]['position']) - np.array(ownship['position'])
        ang = angle360(math.degrees(np.arctan2(ang[1], ang[0])))

        angdiff = ang - ownship['heading']
        if angdiff < -180:
            angdiff += 360
        elif angdiff > 180:
            angdiff -= 360
        if angdiff > 0:
            turn_rate = random.uniform(90.0, ownship['turn_rate_range'][1])
        else:
            turn_rate = random.uniform(ownship['turn_rate_range'][0], -90.0)

        # もし最近接が前にいたら後退

        if abs(angdiff) < 30 and dist_closest < 50:
            thrust = random.uniform(ownship['thrust_range'][0], 0.0)
        elif abs(angdiff) > 150:
            thrust = random.uniform(0.0, ownship['thrust_range'][1])
        else:
            thrust = random.uniform(ownship['thrust_range'][0], ownship['thrust_range'][1])
        fire_bullet = abs(angdiff) < 10.0


        speed = math.sqrt(ownship['velocity'][0] ** 2 + ownship['velocity'][1] ** 2)
        veloangle = math.degrees(math.atan2(ownship['velocity'][1], ownship['velocity'][0]))
        # print(ast_angle(self, ownship['position'], ast_list[closest]['position']))
        if speed > 0:
            if ownship['velocity'][1] > 0:
                travel_angle = -1 * math.degrees(math.asin(ownship['velocity'][0] / speed))
            elif ownship['velocity'][0] > 0:
                travel_angle = -180 + math.degrees(math.asin(ownship['velocity'][0] / speed))
            elif ownship['velocity'][0] < 0:
                travel_angle = 180 + math.degrees(math.asin(ownship['velocity'][0] / speed))
            else:
                travel_angle = 0


        #search_list = [ast_list[num] for num in np.where(np.array(dist_list) < search_range)[0]]
        #search_dist = [dist for dist in dist_list if dist < search_range]
        sorteddict = sorted(ast_list, key=lambda x:math.dist(ownship['position'], x['position']))
        search_list = sorteddict[0:5]
        search_dist = np.array([math.dist(ownship['position'], ast['position']) for ast in search_list])
        vertical = ast_list[closest]['position'][1] - ownship['position'][1]
        horizontal = ast_list[closest]['position'][0] - ownship['position'][0]
        hyp2 = dist_list[closest]
        vx_mult = ast_list[closest]['velocity'][0]
        vy_mult = ast_list[closest]['velocity'][1]
        bloop = 0
        shipang = np.array(ownship['position'])
        Risk = np.zeros(len(dist_list))

        for i in range(len(search_list)):
            ang = np.array(ast_list[i]['position']) - shipang
            ang = angle360(math.degrees(np.arctan2(ang[1], ang[0])))
            angdiff = ang - ownship['heading']
            if angdiff < -180:
                angdiff += 360
            elif angdiff > 180:
                angdiff -= 360
            self.Con1_sim.input['A1'] = search_dist[i]
            self.Con1_sim.input['B1'] = abs(angdiff)
            self.Con1_sim.compute()
            Risk[i] = self.Con1_sim.output['Con1']
        Target = np.argmax(Risk)
        print(Target)

        ang = np.array(search_list[Target]['position']) - np.array(ownship['position'])
        ang = angle360(math.degrees(np.arctan2(ang[1], ang[0])))
        angdiff = ang - ownship['heading']
        if angdiff < -180:
            angdiff += 360
        elif angdiff > 180:
            angdiff -= 360

        if angdiff > 0:
            turn_rate = random.uniform(90.0, ownship['turn_rate_range'][1])
        else:
            turn_rate = random.uniform(ownship['turn_rate_range'][0], -90.0)
        if abs(angdiff) < 30 and dist_closest < 50:
            thrust = random.uniform(ownship['thrust_range'][0], 0.0)
        elif abs(angdiff) > 150:
            thrust = random.uniform(0.0, ownship['thrust_range'][1])
        else:
            thrust = random.uniform(ownship['thrust_range'][0], ownship['thrust_range'][1])
            # fire_bullet = random.uniform(0.45, 1.0) < 0.5
        fire_bullet = abs(angdiff) < 10.0

        # 前後，回転，射撃のタプルをリターンする(thrust…±480m/s^2 turn_rate…±180/s)
        return (thrust, turn_rate, fire_bullet)

    @property
    def name(self) -> str:
        return "OMU-CILab1"
