"""
LQR steering control using the kinematic model in the body frame. See [A Tutorial On Autonomous Vehicle Steering Controller Design, Simulation and Implementation](https://arxiv.org/abs/1803.03758)
"""


import os
import sys

import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

try:
    from Planning.Curve.reeds_shepp import ReedsShepp as RS
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                    "/../../")
    from Planning.Curve.reeds_shepp import ReedsShepp as RS
try:
    import draw
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import draw



class Param:
    # System config
    dt = 0.1
    dist_stop = 0.5

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 1.165 * 2  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width
    MAX_STEER = np.deg2rad(40)  # [rad]
    MAX_ACC = 5.0  # [m / s^2]
    MAX_SPEED = 35 / 3.6  # [m / s]

    # controller config
    Q = np.diag([0.5, 0.5])
    R = np.diag([1.0])

    Kp = 0.3

    # lqr solver config
    max_iteration = 150
    eps = 0.01


class Gear:
    GEAR_DRIVE = 1
    GEAR_REVERSE = 2


def wrap_angle(theta):
    """Limit angle in [-pi, pi)
    """
    theta = theta % (2*math.pi)
    if theta >= math.pi:
        theta -= 2*math.pi
    return theta


class Node:
    """Vehicle state and dynamics
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0,
                 v=0.0, gear=Gear.GEAR_DRIVE):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.gear = gear

    def update(self, delta, a, gear=Gear.GEAR_DRIVE):
        """
        update states of vehicle
        Args:
            delta: steering angle [rad]
            a: acceleration [m / s^2]
            gear: gear mode [GEAR_DRIVE / GEAR/REVERSE]
        """

        wheelbase_ = Param.WB
        delta, a = self.RegulateInput(delta, a)
        dt = Param.dt
        self.gear = gear
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / wheelbase_ * math.tan(delta) * dt

        if gear == Gear.GEAR_DRIVE:
            self.v += a * dt
        else:
            self.v += -1.0 * a * dt

        self.v = self.RegulateOutput(self.v)

    @staticmethod
    def RegulateInput(delta, a):
        delta = np.clip(delta, -Param.MAX_STEER, Param.MAX_STEER)
        a = np.clip(a, -Param.MAX_ACC, Param.MAX_ACC)
        return delta, a

    @staticmethod
    def RegulateOutput(v):
        v = np.clip(v, -Param.MAX_SPEED, Param.MAX_SPEED)
        return v


class TrajectoryAnalyzer:
    def __init__(self, x, y, yaw, k):
        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw
        self.k_ = k

        self.ind_old = 0
        self.ind_end = len(x)

    def calc_lateral_error(self, node):
        """
        errors to trajectory frame
        theta_e = yaw_vehicle - yaw_ref_path
        e_cg = lateral distance of center of gravity (cg) in frenet frame
        """
        x, y, yaw = node.x, node.y, node.yaw
        ind = self.nearest_index(node)

        # calc lateral relative position of vehicle to ref path
        yaw_ref = self.yaw_[ind]
        t_hat = np.array([np.cos(yaw_ref), np.sin(yaw_ref)])
        d = np.array([x-self.x_[ind], y-self.y_[ind]])
        e_cg = -d[0]*t_hat[1] + d[1]*t_hat[0]

        # calc yaw error: theta_e = yaw_vehicle - yaw_ref
        yaw_ref = self.yaw_[ind]
        theta_e = wrap_angle(yaw - yaw_ref)

        # calc ref curvature
        k_ref = self.k_[self.ind_old]
        self.ind_old = ind
        return theta_e, e_cg, yaw_ref, k_ref

    def nearest_index(self, node):
        """
        find the index of the nearest point to current position.
        :param node: current information
        :return: nearest index
        """        
        # calc nearest point in ref path
        dx = [node.x - ix for ix in self.x_[self.ind_old: self.ind_end]]
        dy = [node.y - iy for iy in self.y_[self.ind_old: self.ind_end]]

        ind_add = np.argmin(np.hypot(dx, dy))

        return self.ind_old + ind_add


class LatController:
    """
    Lateral Controller using LQR
    """
    state_size = 2
    control_size = 1

    def ComputeControlCommand(self, node, ref_trajectory):
        """
        calc lateral control command.
        Args:
            node: vehicle state
            ref_trajectory: reference trajectory (analyzer)

        Returns: 
            steer_angle
        """

        dt = Param.dt

        theta_e, e_cg, yaw_ref, k_ref = \
            ref_trajectory.calc_lateral_error(node)

        matrix_ad_, matrix_bd_ = self.UpdateMatrix(node)

        matrix_state_ = np.zeros((self.state_size, 1))
        matrix_state_[0, 0] = e_cg
        matrix_state_[1, 0] = theta_e

        matrix_q_ = Param.Q
        matrix_r_ = Param.R
        matrix_k_ = self.SolveLQRProblem(
            matrix_ad_, matrix_bd_, matrix_q_, matrix_r_, Param.eps, Param.max_iteration)

        steer_angle_feedback = -(matrix_k_ @ matrix_state_)[0][0]

        steer_angle_feedforward = self.ComputeFeedForward(k_ref)

        steer_angle = steer_angle_feedback + steer_angle_feedforward
        return steer_angle

    @staticmethod
    def ComputeFeedForward(ref_curvature):
        """
        LQR control systems linearized at operation point, $u_e = u - u_f = k X $, so $u = u_f + k X$, $u_f$ is the feed forward term
        """
        wheelbase_ = Param.WB
        steer_angle_feedforward = math.atan(wheelbase_ * ref_curvature)
        return steer_angle_feedforward

    @staticmethod
    def SolveLQRProblem(A, B, Q, R, tolerance, max_num_iteration):
        """
        iteratively calculating feedback matrix K
        :param A: matrix_a_
        :param B: matrix_b_
        :param Q: matrix_q_
        :param R: matrix_r_
        :param tolerance: lqr_eps
        :param max_num_iteration: max_iteration
        :return: feedback matrix K
        """

        assert np.size(A, 0) == np.size(A, 1) and \
            np.size(B, 0) == np.size(A, 0) and \
            np.size(Q, 0) == np.size(Q, 1) and \
            np.size(Q, 0) == np.size(A, 1) and \
            np.size(R, 0) == np.size(R, 1) and \
            np.size(R, 0) == np.size(B, 1), \
            "LQR solver: one or more matrices have incompatible dimensions."

        M = np.zeros((np.size(Q, 0), np.size(R, 1)))

        AT = A.T
        BT = B.T
        MT = M.T

        P = Q
        num_iteration = 0
        diff = math.inf

        while num_iteration < max_num_iteration and diff > tolerance:
            num_iteration += 1
            P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

            # check the difference between P and P_next
            diff = (abs(P_next - P)).max()
            P = P_next

        if num_iteration >= max_num_iteration:
            print("LQR solver cannot converge to a solution",
                  "last consecutive result diff is: ", diff)

        K = np.linalg.inv(BT @ P @ B + R) @ (BT @ P @ A + MT)

        return K

    @staticmethod
    def UpdateMatrix(node):
        """
        calc A and B matrices of linearized, discrete system.
        """

        dt = Param.dt
        wheelbase_ = Param.WB

        v = node.v

        matrix_ad_ = np.array(
            [[0, v],
             [0, 0]], dtype=float
        )  # time discrete A matrix

        matrix_ad_ = matrix_ad_ * dt + np.eye(2)

        matrix_bd_ = np.array([0.0, v / wheelbase_]).reshape((2, 1))
        matrix_bd_ = dt * matrix_bd_

        return matrix_ad_, matrix_bd_


class LonController:
    """
    Longitudinal Controller using PID.
    """

    @staticmethod
    def ComputeControlCommand(target_speed, node, dist):
        """
        calc acceleration command using PID.
        :param target_speed: target speed [m / s]
        :param node: vehicle state
        :param dist: distance to goal [m]
        :return: control command (acceleration) [m / s^2]
        """

        if node.gear == Gear.GEAR_DRIVE:
            direct = 1.0
        else:
            direct = -1.0

        a = Param.Kp * (target_speed - direct * node.v)

        if dist < 10.0:
            if node.v > 2.0:
                a = -3.0
            elif node.v < -2:
                a = -1.0

        return a


def generate_path(s):
    """
    design path using reeds-shepp path generator.
    divide paths into sections, in each section the direction is the same.
    Args: 
        s: list of [x,y,yaw]

    Returns:
        paths (list): list of path segments
    """
    wheelbase_ = Param.WB

    max_c = math.tan(Param.MAX_STEER/2) / wheelbase_
    x_ref, y_ref, yaw_ref, direct, curv = [], [], [], [], []

    for i in range(len(s) - 1):
        s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

        rs = RS(s_x, s_y, s_yaw, g_x, g_y, g_yaw, max_c, 0.2)
        opt_path = rs.get_optimal_path()
        path_i = rs.slice_path(opt_path)

        for j in range(len(path_i)):
            if len(x_ref) == 0:
                x_ref.append(list(path_i[j][:, 0]))
                y_ref.append(list(path_i[j][:, 1]))
                yaw_ref.append(list(path_i[j][:, 2]))
                curv.append(list(path_i[j][:, 3]))
                direct.append(list(path_i[j][:, 4]))
            elif direct[-1][-1] == path_i[j][0, -1]:
                x_ref[-1] += path_i[j][:, 0].tolist()
                y_ref[-1] += path_i[j][:, 1].tolist()
                yaw_ref[-1] += path_i[j][:, 2].tolist()
                curv[-1] += path_i[j][:, 3].tolist()
                direct[-1] += path_i[j][:, 4].tolist()
            else:
                x_ref.append(list(path_i[j][:, 0]))
                y_ref.append(list(path_i[j][:, 1]))
                yaw_ref.append(list(path_i[j][:, 2]))
                curv.append(list(path_i[j][:, 3]))
                direct.append(list(path_i[j][:, 4]))

    return x_ref, y_ref, yaw_ref, direct, curv

import imageio

def main():
    # generate path
    states = [(0, 0, 0), (20, 30, 0), (35, 20, 90), (40, 0, 180),
              (20, 0, 180), (20, 10, 180), (20, 15, 180)]

    x_ref, y_ref, yaw_ref, direct, curv = generate_path(states)

    x_all, y_all = [], []
    for i in range(len(x_ref)):
        x_all += list(x_ref[i])
        y_all += list(y_ref[i])

    wheelbase_ = Param.WB

    maxTime = 100.0

    x0, y0, yaw0, direct0 = \
        x_ref[0][0], y_ref[0][0], yaw_ref[0][0], direct[0][0]

    x_rec, y_rec, yaw_rec, direct_rec = [], [], [], []

    lat_controller = LatController()
    lon_controller = LonController()

    dt = Param.dt
    imgs = []
    for x, y, yaw, gear, k in zip(x_ref, y_ref, yaw_ref, direct, curv):
        t = 0.0

        ref_trajectory = TrajectoryAnalyzer(x, y, yaw, k)
        direct0 = gear[0]
        node = Node(x=x0, y=y0, yaw=yaw0, v=0.1, gear=direct0)

        while t < maxTime:

            dist = math.hypot(node.x - x[-1], node.y - y[-1])

            if direct0 == Gear.GEAR_DRIVE:
                target_speed = 25.0 / 3.6
            else:
                target_speed = 15.0 / 3.6

            delta_opt = lat_controller.ComputeControlCommand(
                node, ref_trajectory)

            a_opt = lon_controller.ComputeControlCommand(
                target_speed, node, dist)

            node.update(delta_opt, a_opt, direct0)
            ind = ref_trajectory.ind_old
            t += dt

            x_rec.append(node.x)
            y_rec.append(node.y)
            yaw_rec.append(node.yaw)

            x0 = x_rec[-1]
            y0 = y_rec[-1]
            yaw0 = yaw_rec[-1]

            if dist <= 0.5:
                break

            plt.cla()
            plt.plot(x_all, y_all, color='gray', linewidth=2.0)
            plt.plot(x_rec, y_rec, linewidth=2.0, color='darkviolet')
            draw.draw_car(node.x, node.y, node.yaw, delta_opt, Param)
            plt.plot([x[ind]], [y[ind]], '.r')
            
            plt.axis("equal")
            plt.title("LQR (kinematic): v=" +
                      str(node.v * 3.6)[:4] + "km/h")
            fname = 'tmp.png'
            plt.savefig(fname)
            imgs.append(plt.imread(fname))
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event:
                                         [exit(0) if event.key == 'escape' else None])
            plt.pause(0.001)
    imageio.mimsave('lqr_kinematic.gif', imgs,duration=Param.dt)
    plt.show()


if __name__ == '__main__':
    main()
