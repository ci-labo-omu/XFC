import math
from typing import Tuple, Dict, Any, List
import random
from src.kessler_game.controller import KesslerController
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from func import aim, ast_angle, angle360, aiming_function
from src.kessler_game.ship import Ship


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
        self.wack = 0

        individual = [599, 486, 157, 104, 243, 59, 522, 521, 83, 872, 191, 71, 862, 215, 439, 274, 719, 960, 700, 664,
                      74, 624, 651, 176, 547, 747, 251, 168, 474, 389, 277, 948, 656, 705, 571, 225, 542, 918, 466, 787,
                      795, 291, 235, 615]
        #              1    2    3    4    5    6    7   8    9   10   11   12  13   14   15   16   17   18   19   20   21  22   23    24  25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41   42   43   44

        """
        Gene Constants
        """
        self.roe_zone = individual[36] / 1000 * 500  # 112.5
        self.roe_zone = 610 / 1000 * 500  # 305
        # Maximum Distance for Multitasking, Default: 240
        self.fuzzy_roe = individual[37] / 1000 * self.roe_zone  #
        self.fuzzy_roe = 918 / 1000 * self.roe_zone
        # Minimum Distance for Fuzzy application, Default: 120
        self.edgeclear = 542 / 1000 * 275
        # determines the distance from the edge before edge clearing behavior starts
        self.wack_coef = individual[38] / 1000 * 200
        self.wack_coef = 230
        # Controls Rate of Fire, considering distance. 10 is 1 per target, Default: 100
        self.brake_speed_power = individual[39] / 1000 * 500
        # Controls Speed at which the craft will not exceed. Variable by size. Default: 250, lower is faster.
        self.SAD_base = individual[40] / 1000 * 100
        # S&D closeness to target, base. Default: 50
        self.SAD_size_adjust = individual[41] / 1000 * 100
        # S&D closeness to target, varies with size. Default: 62
        self.evasive_base = individual[42] / 1000 * 100
        # Closeness where evasive behavior is triggered. Default: 45
        self.evasive_size_adjust = individual[43] / 1000 * 50
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

        #テストとしてTeam Asimovから引用
        names = ['near', 'close', 'far']
        A1 = ctrl.Antecedent(np.arange(-91, 271, 1), 'A1')
        A1.automf(names = names)

        B1 = ctrl.Antecedent(np.arange(-91, 271, 1), 'B1')
        B1['front'] = fuzz.trimf(B1.universe, [-15, 0, 15])
        B1['middle'] = fuzz.trimf(B1.universe, [15, 45, 75])
        B1['back'] = fuzz.trimf(B1.universe, [75, 180, 285])


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
        B2= ctrl.Antecedent(np.arange(-91, 271, 1), 'B2')
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


        def r_angle(opposite, hypotenuse, abovebelow, leftright):
            if abovebelow > 0:
                angle = -1 * (math.degrees(math.asin(opposite / hypotenuse)))
            elif abovebelow < 0 and leftright < 0:
                angle = 180 + (math.degrees(math.asin(opposite / hypotenuse)))
            elif abovebelow < 0 and leftright > 0:
                angle = -180 + (math.degrees(math.asin(opposite / hypotenuse)))
            else:
                angle = 0
            return angle

        self.rangle = r_angle





    def actions(self, ownship: Ship, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        ship_list = input_data["ships"]
        ast_list = input_data["asteroids"]
        dist_list = [math.dist(ownship['position'], ast['position']) for ast in ast_list]
        closest = np.argmin(dist_list)
        ang = np.array(ast_list[closest]['position']) - np.array(ownship['position'])
        ang = angle360(math.degrees(np.arctan2(ang[1], ang[0])))
        print("あんぐ")
        print(ang)
        print(ownship['heading'])
        angdiff = ang - ownship['heading']
        if angdiff < -180: angdiff += 360
        elif angdiff > 180: angdiff -= 360

        """print(angdiff)
        if angdiff > 0: turn_rate = random.uniform(0.0, ownship['turn_rate_range'][1])
        else: turn_rate = random.uniform( ownship['turn_rate_range'][0], 0.0)
        """
        #もし最近接が前にいたら後退
        """
        if abs(angdiff) < 30:
            thrust = random.uniform(ownship['thrust_range'][0], 0.0)
        elif abs(angdiff)>150:
            thrust = random.uniform(0.0, ownship['thrust_range'][1])
        else: thrust = random.uniform(ownship['thrust_range'][0], ownship['thrust_range'][1])"""
        #fire_bullet = random.uniform(0.45, 1.0) < 0.5
        fire_bullet = abs(angdiff) < 15
        print(fire_bullet)

        speed = math.sqrt(ownship['velocity'][0]**2 + ownship['velocity'][1]**2)
        veloangle = math.degrees(math.atan2(ownship['velocity'][1], ownship['velocity'][0]))
        #print(ast_angle(self, ownship['position'], ast_list[closest]['position']))
        if speed > 0:
            if ownship['velocity'][1] > 0:
                travel_angle = -1 * math.degrees(math.asin(ownship['velocity'][0] / speed))
            elif ownship['velocity'][0] > 0:
                travel_angle = -180 + math.degrees(math.asin(ownship['velocity'][0] / speed))
            elif ownship['velocity'][0] < 0:
                travel_angle = 180 + math.degrees(math.asin(ownship['velocity'][0] / speed))
            else:
                travel_angle = 0



        sidefromcenter = 400 - ownship['position'][0]
        above_center = 300 - ownship['position'][1]
        distancefromcenter = ((ownship['position'][0] - 400) ** 2 + (ownship['position'][1] - 300) ** 2) ** 0.5
        if distancefromcenter > 0:
            anglefromcenter = self.rangle(sidefromcenter, distancefromcenter, above_center, sidefromcenter)
        else:
            anglefromcenter = 0

        ab2 =ast_list[closest]['position'][1] - ownship['position'][1]
        lr2 = ast_list[closest]['position'][0] - ownship['position'][0]
        op2 = lr2
        hyp2 = dist_list[closest]
        s_rangle_inrange = self.rangle(op2, hyp2, ab2, lr2)
        astangle = angle360(s_rangle_inrange)
        vx_mult = ast_list[closest]['velocity'][0]
        vy_mult = ast_list[closest]['velocity'][1]
        normal_astangle_mult = angle360(s_rangle_inrange)
        ant_angle2 = aiming_function(vx_mult, vy_mult, normal_astangle_mult)
        orientation2 = abs(ownship['heading'] - s_rangle_inrange + ant_angle2)
        bloop = 0

        if ast_list[closest]['size'] == 4:
            self.Con4_sim.input['A4'] = orientation2
            self.Con4_sim.input['B4'] = dist_list[closest]
            self.Con4_sim.compute()
            Favorability = self.Con4_sim.output['Con4']
        elif ast_list[closest]['size'] == 3:
            self.Con3_sim.input['A3'] = orientation2
            self.Con3_sim.input['B3'] = dist_list[closest]
            self.Con3_sim.compute()
            Favorability = self.Con3_sim.output['Con3']
        elif ast_list[closest]['size'] == 2:
            self.Con2_sim.input['A2'] = orientation2
            self.Con2_sim.input['B2'] = dist_list[closest]
            self.Con2_sim.compute()
            Favorability = self.Con2_sim.output['Con2']
        elif ast_list[closest]['size'] == 1:
            self.Con1_sim.input['A1'] = orientation2
            self.Con1_sim.input['B1'] = dist_list[closest]
            self.Con1_sim.compute()
            Favorability = self.Con1_sim.output['Con1']
        if Favorability > 0:
                Target = closest
                bloop = bloop + 1

        abovebelow = input_data['asteroids'][closest]['position'][1] - ownship['position'][1]
        leftright = input_data['asteroids'][closest]['position'][0] - ownship['position'][0]
        opposite = (input_data['asteroids'][closest]['position'][0] - ownship['position'][0])
        hypotenuse = dist_list[closest]
        s_rangle = self.rangle(opposite, hypotenuse, abovebelow, leftright)
        orientation = abs(ownship['heading'] - s_rangle)
        if bloop > 0:
            if ast_list[closest]['size'] < 4 and dist_list[closest] > 103:
                Target_orientation = orientation2
                Target_angle = s_rangle_inrange
                Target_Distance = dist_list[closest]
                # Target_size = inrange_size[Target]
                # Target_Favorability = Favorability[m]

            else:
                Target_orientation = orientation
                Target_angle = s_rangle
                Target_Distance = dist_list[closest]
                # Target_size = input_data['asteroids'][closest_asteroid]['size']
                # Target_Favorability = 0
        elif bloop == 0:
            Target_orientation = orientation
            Target_angle = s_rangle
            Target_Distance = dist_list[closest]
            # Target_size = input_data['asteroids'][closest_asteroid]['size']
            # Target_Favorability = 0

        """
        s_rangle is the angle relative to the ship necessary for the ship to point at the closest asteroid
        """
        """ Positive if above, negative if below"""
        """ negative if left, positive if right"""

        normal_shipangle = angle360(ownship['heading'])
        normal_astangle = angle360(s_rangle)
        normal_cangle = angle360(anglefromcenter)
        normal_target_angle = angle360(Target_angle)
        clast_size = ast_list[closest]['size']
        dodge_counter = 0
        if Target_orientation == orientation:
            vx_mult = input_data['asteroids'][closest]['velocity'][0]
            vy_mult = input_data['asteroids'][closest]['velocity'][1]

            self.ant_angle = aiming_function(vx_mult, vy_mult, normal_astangle)
            Target_orientation = abs(ownship['heading'] - s_rangle + self.ant_angle)
            leftright_target = self.leftright(normal_shipangle, normal_target_angle)
        else:
            vx_mult = ast_list[closest]['velocity'][0]
            vy_mult = ast_list[closest]['velocity'][1]
            Target_s_rangle = s_rangle_inrange
            Target_normal_astangle = angle360(Target_s_rangle)
            self.ant_angle = aiming_function(vx_mult, vy_mult, Target_normal_astangle)
            Target_orientation = abs(ownship['heading'] - Target_s_rangle + self.ant_angle)
            leftright_target = self.leftright(normal_shipangle, normal_target_angle)
        vx_mult = ast_list[closest]['velocity'][0]
        vy_mult = ast_list[closest]['velocity'][1]
        self.ant_angle = aiming_function(vx_mult, vy_mult, normal_astangle)
        leftright_dodge = self.leftright(normal_shipangle, normal_astangle)
        """
        This is the master if function in which it determines which behavior mode to fall into 
        """

        if  dist_list[closest] < 45 + (5 * clast_size):  # Respawn Behavior
            if orientation > 160:
                thrust = ownship['thrust_range'][1]
            elif orientation <= 160:
                thrust = 0
                turn_rate = 180

        else:

            if speed > 1 + (dist_list[closest] / self.brake_speed_power):  # Braking Speed Determinant

                """
                Braking Manuever- For if the ship is going to fast. Probably best for when there's a lot of 
                asteroids and you do you don't want it to slignshot past on into another
                """

                t_orientation = abs(ownship['heading'] - travel_angle)
                print('t_orientation')
                print(t_orientation)
                if travel_angle == 0:
                    pass
                elif t_orientation > 60:
                    thrust = ownship['thrust_range'][1]
                elif t_orientation < 60:
                    thrust = ownship['thrust_range'][0]
                else:
                    print('something wonky afoot')

            elif dist_list[closest] < self.evasive_base + (self.evasive_size_adjust * clast_size):
                """Evasive Manuevers, I think we could expand this to considering the closest three 
                    asteroids and fuzzily deciding which direction to flee in
                    for cases where an asteroid is perpindicularly approaching it needs to be able to distinguish left and right anf
                    behave accordingly """
                dodge_counter = 1
                if orientation > 90:
                    thrust = ownship['thrust_range'][1]
                elif orientation > 90 and orientation < 90:
                    thrust = 0
                else:
                    thrust = ownship['thrust_range'][0]
                print(leftright_dodge)
                print('leftright_dodge')

                if leftright_dodge == 0 or leftright_dodge == 1:
                    if leftright_dodge == 0 and orientation > 1:
                        turn_rate = 180
                    elif leftright_dodge == 0 and orientation <= 1:
                        turn_rate = 90
                    elif leftright_dodge == 1 and orientation > 1:
                        turn_rate = -180
                    else:
                        turn_rate = -90

            elif ownship['position'][0] > 800 - self.edgeclear or ownship['position'][0] < self.edgeclear or ownship['position'][1] > 600 - self.edgeclear or ownship['position'][1] < self.edgeclear:
                turn = self.leftright(normal_shipangle, normal_cangle)
                center_orientation = abs(ownship['heading'] - anglefromcenter)
                if center_orientation < 150:
                    thrust = ownship['thrust_range'][1]
                elif turn == 0:
                    turn_rate = 180
                else:
                    turn_rate = -180

                """
                # Search and Destroy
                elif shortest_distance > 50 + (62 * clast_size):
                ship.thrust = ship.thrust_range[1]
                """
            # Search and Destroy Also Aiming for Target, will probs give it aiming for dodging
            elif dist_list[closest] > self.SAD_base + (self.SAD_size_adjust * clast_size) and dodge_counter == 0:
                thrust = ownship['thrust_range'][1]

            if leftright_target == 0 or leftright_target == 1:
                if leftright_target == 0 and Target_orientation > 3:
                    turn_rate = 180
                elif leftright_target == 0 and Target_orientation <= 3:
                    turn_rate = 90
                elif leftright_target == 1 and Target_orientation > 3:
                    turn_rate = -180
                else:
                    turn_rate = -90

            """
            Shooting Mechanism
            """
            self.wack += self.wack_coef  # wack increases until it reaches a fire threshold


            if dodge_counter == 0:
                if orientation < 3 * clast_size or orientation2 < 3:
                    if self.wack > Target_Distance:
                        self.wack = 0
                        fire_bullet = 1
            else:
                if Target_orientation < 3 * clast_size or orientation2 < 3:
                    if self.wack > Target_Distance:
                        self.wack = 0
                        fire_bullet = 1

        #前後，回転，射撃のタプルをリターンする
        return (thrust, turn_rate, fire_bullet)

    @property
    def name(self) -> str:
        return "OMU-CILab1"
