import math
from typing import Tuple, Dict
from kesslergame import KesslerController
import numpy as np
from func import angle360
from kesslergame import Ship


class NewController3(KesslerController):
    """
         A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
        game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
        game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
        """

    def __init__(self, gene, genes2=None):
        """
        Create your fuzzy logic controllers and other objects here
        """
        # gene: [left, center, right, (angle)center]
        left1 = gene[0]
        left2 = gene[1]
        center = gene[2]
        right2 = gene[3]
        right1 = gene[4]
        center_angle = gene[5]
        def membership1(x):
            if x <= left1:
                return [1.0, 0.0, 0.0, 0.0, 0.0]
            elif x <= left2:
                return np.array([1.0 - (x - left1) / (left2 - left1), (x - left1) / (left2 - left1), 0.0, 0.0, 0.0])
            elif x <= center:
                return np.array([0.0, (center - x) / (center - left2),
                                 (x - left2) / (center - left2), 0.0, 0.0])
            elif x <= right2:
                return np.array([0.0, 0.0, (right2 - x) / (right2 - center), (x - center) / (right2 - center), 0.0])
            elif x <= right1:
                return np.array([0.0, 0.0, 0.0, (right1 - x) / (right1 - right2), (x - right2) / (right1 - right2)])
            else:
                return np.array([0.0, 0.0, 0.0, 0.0, 1.0])

        def membership2(angle):
            angle = abs(angle)
            if angle <= center_angle:
                return np.array([1.0 - angle / center_angle, angle / center_angle, 0.0])
            elif angle <= 180:
                return np.array([0.0, 2 - angle / center_angle, angle / center_angle - 1])

        self.membership1 = membership1
        self.membership2 = membership2

        def mems(x, angle):
            Rules = {
                0: "x1 is near  and x2 is front ",
                1: "x1 is near  and x2 is middle",
                2: "x1 is near  and x2 is back",
                3: "x1 is close and x2 is front",
                4: "x1 is close and x2 is middle",
                5: "x1 is close and x2 is back",
                6: "x1 is far   and x2 is front",
                7: "x1 is far   and x2 is middle",
                8: "x1 is far   and x2 is back",
            }

            # out0 = np.array([-480, 90])
            # out1 = np.array([-360, 180])
            # out2 = np.array([120, 180])
            # out3 = np.array([-360, 180])
            # out4 = np.array([-60, 180])
            # out5 = np.array([180, 180])
            # out6 = np.array([120, 180])
            # out7 = np.array([120, 180])
            # out8 = np.array([0, 180])

            out0 = np.array([genes2[0], genes2[15]])
            out1 = np.array([genes2[1], genes2[16]])
            out2 = np.array([genes2[2], genes2[17]])
            out3 = np.array([genes2[3], genes2[18]])
            out4 = np.array([genes2[4], genes2[19]])
            out5 = np.array([genes2[5], genes2[20]])
            out6 = np.array([genes2[6], genes2[21]])
            out7 = np.array([genes2[7], genes2[22]])
            out8 = np.array([genes2[8], genes2[23]])
            out9 = np.array([genes2[9], genes2[24]])
            out10 = np.array([genes2[10], genes2[25]])
            out11 = np.array([genes2[11], genes2[26]])
            out12 = np.array([genes2[12], genes2[27]])
            out13 = np.array([genes2[13], genes2[28]])
            out14 = np.array([genes2[14], genes2[29]])

            k = membership1(x)
            p = membership2(angle)

            rule0 = k[0] * p[0]
            rule1 = k[0] * p[1]
            rule2 = k[0] * p[2]
            rule3 = k[1] * p[0]
            rule4 = k[1] * p[1]
            rule5 = k[1] * p[2]
            rule6 = k[2] * p[0]
            rule7 = k[2] * p[1]
            rule8 = k[2] * p[2]
            rule9 = k[3] * p[0]
            rule10 = k[3] * p[1]
            rule11 = k[3] * p[2]
            rule12 = k[4] * p[0]
            rule13 = k[4] * p[1]
            rule14 = k[4] * p[2]

            out = ((rule0 * out0) + (rule1 * out1) + (rule2 * out2) + (rule3 * out3) + (rule4 * out4) + (
                    rule5 * out5) + (rule6 * out6) + (
                           rule7 * out7) + (rule8 * out8) + (rule9 * out9) + (rule10 * out10) + (rule11 * out11) + (
                               rule12 * out12) + (rule13 * out13)+(rule14*out14))
            return out

        self.mems = mems

        center_x = 500
        center_y = 400

    def actions(self, ownship: Dict, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        # 隕石と機体の位置関係のセクション
        ast_list = np.array(input_data["asteroids"])
        dist_xylist = [np.array(ownship['position']) - np.array(ast['position']) for ast in ast_list]
        dist_avoidlist = dist_xylist.copy()
        dist_list1 = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_xylist]
        closest = np.argmin(dist_list1)
        dist_closest1 = dist_list1[closest]

        # よける部分に関しては画面端のことを考える，弾丸はすり抜けないから狙撃に関しては考えない
        sidefromcenter = 500 - ownship['position'][0]
        below_center = 400 - ownship['position'][1]
        for xy in dist_avoidlist:
            if xy[0] > 500:
                xy[0] -= 1000
            elif xy[0] < -500:
                xy[0] += 1000
            if xy[1] > 400:
                xy[1] -= 800
            elif xy[1] < -400:
                xy[1] += 800
        dist_avoidlist = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_avoidlist]

        sorted2_idx = np.argsort(dist_avoidlist)
        sorteddict = ast_list[sorted2_idx]
        search_list = sorteddict[0:5]
        search_dist = np.array([math.dist(ownship['position'], ast['position']) for ast in search_list])
        near_angle = [np.array(ast['position']) - np.array(ownship['position']) for ast in search_list]
        near_angle = [angle360(math.degrees((np.arctan2(near_ang[1], near_ang[0])))) - ownship['heading'] for near_ang
                      in near_angle]
        aalist = []
        for ang in near_angle:
            if ang > 180:
                ang -= 360
            elif ang < -180:
                ang += 360
            aalist.append(ang)

        angdiff_front = min(aalist, key=abs)
        angdiff = aalist[np.argmin(search_dist)]
        fire_bullet = abs(angdiff_front) < 10 and min(dist_list1) < 400
        avoidance = np.min(dist_avoidlist)
        """if len(input_data['ships']) >= 2:
            dist = math.dist(ownship['position'], input_data['ships'][1-ownship['id']]['position'])
            if dist < 20:
                avoidance = dist"""
        rule = self.mems(avoidance, angdiff)

        thrust = rule[0]
        turn_rate = rule[1] * np.sign(angdiff)

        # Team1がよける

        if thrust > ownship['thrust_range'][1]:
            thrust = ownship['thrust_range'][1]
        elif thrust < ownship['thrust_range'][0]:
            thrust = ownship['thrust_range'][0]
        if turn_rate > ownship['turn_rate_range'][1]:
            turn_rate = ownship['turn_rate_range'][1]
        elif turn_rate < ownship['turn_rate_range'][0]:
            turn_rate = ownship['turn_rate_range'][0]
        # 前後，回転，射撃のタプルをリターンする(thrust…±480m/s^2 turn_rate…±180/s)
        return thrust, turn_rate, fire_bullet

    @property
    def name(self) -> str:
        return "Optimized Genes"
