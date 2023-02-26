import math
from typing import Tuple, Dict
from kesslergame import KesslerController
import numpy as np
from func import angle360
from kesslergame import Ship
# ルール要素に速さを追加したい

class NewController2(KesslerController):
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
        left = gene[0]
        center = gene[1]
        right = gene[2]
        center_angle = gene[3]
        left_speed = gene[4]
        center_speed = gene[5]
        right_speed = gene[6]

        def membership1(x):
            if x <= gene[1]:
                return np.array([1.0 - (x - left) / (center - left), (x - left) / (center - left), 0.0])
            elif x <= right:
                return np.array([0.0, right / (right - center) - x / (right - center),
                                 x / (right - center) - center / (right - center)])
            else:
                return np.array([0.0, 0.0, 1.0])

        def membership2(angle):
            angle = abs(angle)
            if angle <= center_angle:
                return np.array([1.0 - angle / center_angle, angle / center_angle, 0.0])
            elif angle <= 180:
                return np.array([0.0, 2 - angle / center_angle, angle / center_angle - 1])

        def membership3(speed):
            if speed <= center_speed:
                return np.array([1.0 - (speed - left_speed) / (center_speed - left_speed), (speed - left_speed) / (center_speed - left_speed), 0.0])
            elif speed <= right_speed:
                return np.array([0.0, right_speed / (right_speed - center_speed) - speed / (right_speed - center_speed),
                                 speed / (right_speed - center_speed) - center_speed / (right_speed - center_speed)])
            else:
                return np.array([0.0, 0.0, 1.0])

        self.membership1 = membership1
        self.membership2 = membership2
        self.center = gene[1]

        def mems(x, angle, speed):
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

            #out0 = np.array([-480, 90])
            #out1 = np.array([-360, 180])
            #out2 = np.array([120, 180])
            #out3 = np.array([-360, 180])
            #out4 = np.array([-60, 180])
            #out5 = np.array([180, 180])
            #out6 = np.array([120, 180])
            #out7 = np.array([120, 180])
            #out8 = np.array([0, 180])
            #output = np.stack([genes2[0:10], genes2[9:19], genes2[18:28]], 1)
            # 1列目が前後加速度，2列目が角速度の絶対値
            out = np.stack([genes2[0:27], genes2[27:54]],1)
            # out0 = np.array([genes2[0], genes2[9]])
            # out1 = np.array([genes2[1], genes2[10]])
            # out2 = np.array([genes2[2], genes2[11]])
            # out3 = np.array([genes2[3], genes2[12]])
            # out4 = np.array([genes2[4], genes2[13]])
            # out5 = np.array([genes2[5], genes2[14]])
            # out6 = np.array([genes2[6], genes2[15]])
            # out7 = np.array([genes2[7], genes2[16]])
            # out8 = np.array([genes2[8], genes2[17]])
            # ...
            # out27= np.array([genes2[26],genes2[53]
            mf_dist = membership1(x)
            mf_angle = membership2(angle)
            mf_speed = membership3(speed)
            rules = np.zeros(27)

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        rules[9*i+3*j+k] = mf_dist[i] * mf_angle[j] * mf_speed[k]

            # rules[0]  = k[0] * p[0] * q[0]
            # rules[1]  = k[0] * p[0] * q[1]
            # rules[2]  = k[0] * p[0] * q[2]
            # rules[3]  = k[0] * p[1] * q[0]
            # rules[4]  = k[0] * p[1] * q[1]
            # rules[5]  = k[0] * p[1] * q[2]
            # rules[6]  = k[0] * p[2] * q[0]
            # rules[7]  = k[0] * p[2] * q[1]
            # rules[8]  = k[0] * p[2] * q[2]
            # ...
            # rules[27] = k[2] * p[2] * q[2]

            # out = ((rule0 * out0) + (rule1 * out1) + (rule2 * out2) + (rule3 * out3) + (rule4 * out4) + (
            #         rule5 * out5) + (rule6 * out6) + (
            #                rule7 * out7) + (rule8 * out8))
            output = np.array([0, 0])
            for i in range(len(out)):
                output += out[i] * rules[i]
            output = rules * out
            return output

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
        avoid_speed = ast_list[np.argmin(dist_avoidlist)]['speed']
        """if len(input_data['ships']) >= 2:
            dist = math.dist(ownship['position'], input_data['ships'][1-ownship['id']]['position'])
            if dist < 20:
                avoidance = dist"""
        rule = self.mems(avoidance, angdiff, avoid_speed)

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
        return "OMU-Let's"
