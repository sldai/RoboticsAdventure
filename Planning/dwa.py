"""
Dynamic Window Approach
https://ieeexplore.ieee.org/abstract/document/580977/
Modified from PythonRobotics
"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class Config:
        max_speed = 1.0  # [m/s]
        min_speed = -0.5  # [m/s]
        max_yaw_rate = 80.0 * math.pi / 180.0  # [rad/s]
        max_accel = 0.2  # [m/ss]
        max_delta_yaw_rate = 80.0 * math.pi / 180.0  # [rad/ss]
        v_resolution = 0.01  # [m/s]
        yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        dt = 0.1  # [s] Time tick for motion prediction
        predict_time = 3.0  # [s]
        to_goal_cost_gain = 0.15
        speed_cost_gain = 1.0
        obstacle_cost_gain = 0.3
        robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        robot_radius = 0.5  # [m] for collision check
        obstacle_radius = 1.0 # [m]

        # obstacles [x(m) y(m), ....]
        ob = np.array([[-1, -1],
                        [0, 2],
                        [4.0, 2.0],
                        [5.0, 4.0],
                        [5.0, 5.0],
                        [5.0, 6.0],
                        [5.0, 9.0],
                        [8.0, 9.0],
                        [7.0, 9.0],
                        [8.0, 10.0],
                        [9.0, 11.0],
                        [12.0, 13.0],
                        [12.0, 12.0],
                        [15.0, 15.0],
                        [13.0, 13.0]
                        ])

def motion(x, u, dt):
    """
    motion model
    """
    x = np.array(x).copy()
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def dwa_control(x, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x)

    u, trajectory = calc_control_and_trajectory(x, dw, goal, ob)

    return u, trajectory


def calc_dynamic_window(x):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [Config.min_speed, Config.max_speed,
          -Config.max_yaw_rate, Config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - Config.max_accel * Config.dt,
          x[3] + Config.max_accel * Config.dt,
          x[4] - Config.max_delta_yaw_rate * Config.dt,
          x[4] + Config.max_delta_yaw_rate * Config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw


def predict_trajectory(x_init, v, yaw_dot):
    """
    predict trajectory with the same input during predction time
    """
    x = np.array(x_init)
    trajectory = [x]
    time = 0
    while time <= Config.predict_time:
        x = motion(x, [v, yaw_dot], Config.dt)
        trajectory.append(x)
        time += Config.dt

    return np.array(trajectory)


def calc_control_and_trajectory(x, dw, goal, ob):
    """
    calculation the best control within the dynamic window
    """
    x_init = np.array(x)

    best_u = [0.0, 0.0]
    best_trajectory = None

    u_samples = []
    traj_samples = []
    cost_samples = []
    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], Config.v_resolution):
        for y in np.arange(dw[2], dw[3], Config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y)
            # calc cost
            to_goal_cost = calc_to_goal_cost(trajectory, goal)
            speed_cost = (Config.max_speed - trajectory[-1, 3])
            ob_cost = calc_obstacle_cost(trajectory, ob)

            cost_samples.append([to_goal_cost,speed_cost,ob_cost])
            u_samples.append([v, y])
            traj_samples.append(trajectory)
    cost_samples = np.array(cost_samples)
    # normalization
    def normalize_0_1(x):
        if np.max(x)-np.min(x)>1e-4:
            return (x-np.min(x))/(np.max(x)-np.min(x))
        else:
            return np.zeros_like(x)

    def normalize_div_sum(x):
        if np.sum(x) > 1e-4:
            return x/np.sum(x)
        else:
            return np.zeros_like(x)

    # for i in range(3):
    #     non_inf = cost_samples[:,i] != np.inf
    #     if np.sum(non_inf) >0:
    #         cost_samples[non_inf,i] = normalize_0_1(cost_samples[non_inf,i])
    
    cost_samples = Config.to_goal_cost_gain*cost_samples[:,0]+Config.speed_cost_gain*cost_samples[:,1]+Config.obstacle_cost_gain*cost_samples[:,2]
    ind = np.argmin(cost_samples)
    best_u = u_samples[ind]
    best_trajectory = traj_samples[ind]

    if abs(best_u[0]) < Config.robot_stuck_flag_cons \
            and abs(x[3]) < Config.robot_stuck_flag_cons:
        # to ensure the robot do not get stuck in
        # [v,w] = [0,0], force the robot rotate
        best_u[1] = -Config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    clearance = np.hypot(dx, dy) 

    if np.array(clearance <= Config.robot_radius).any():
        return np.inf

    min_c = min(np.min(clearance)-Config.robot_radius, 5.0)
    return 1.0 / min_c  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """
    ind = -1
    dx = goal[0] - trajectory[ind, 0]
    dy = goal[1] - trajectory[ind, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[ind, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw):  # pragma: no cover
    yaw = np.linspace(-np.pi, np.pi)
    circle_x = x+np.cos(yaw)*Config.robot_radius
    circle_y = y+np.sin(yaw)*Config.robot_radius
    plt.plot(circle_x, circle_y, '-r')
    # out_x, out_y = (np.array([x, y]) +
    #                 np.array([np.cos(yaw), np.sin(yaw)]) * Config.robot_radius)
    # plt.plot([x, out_x], [y, out_y], "-k")

import imageio

def main(gx=10.0, gy=10.0):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    trajectory = [x]
    ob = Config.ob

    while True:
        u, predicted_trajectory = dwa_control(x, goal, ob)
        x = motion(x, u, Config.dt)  # simulate robot
        trajectory.append(x)  # store state history
        
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2])
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)


        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= Config.robot_radius:
            print("Goal!!")
            done = True
            break
        

    print("Done")
    trajectory = np.array(trajectory)
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
    plt.show()


if __name__ == '__main__':
    main()


