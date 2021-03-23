"""
RRT_CONNECT_2D
Ref: https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
"""

import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")
import env, plotting, utils
from enum import Enum

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None

class ExtendState(Enum):
    reached = 0
    advanced = 1
    trapped = 2

class RRTConnect:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.T1 = [self.s_start]
        self.T2 = [self.s_goal]

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.s_goal, self.goal_sample_rate)
            if self.extend(self.T1, node_rand) != ExtendState.trapped:
                if self.connect(self.T2, self.T1[-1]) == ExtendState.reached:
                    return self.extract_path(self.T1[-1], self.T2[-1])
            if len(self.T2) < len(self.T1):
                list_tmp = self.T2
                self.T2 = self.T1
                self.T1 = list_tmp

        return None

    def extend(self, T, q):
        """Extend T towards q
        Args:
            T (list of nodes)
            q (node)
        """
        q_near = self.nearest_neighbor(T,q)
        q_new = self.new_state(q_near, q)
        if not self.utils.is_collision(q_near, q_new):
            T.append(q_new)
            if self.is_node_same(q,q_new):
                return ExtendState.reached
            else:
                return ExtendState.advanced
        return ExtendState.trapped

    def connect(self, T, q):
        """Connect T towards q
        T : node list
        q : sample node
        """
        while True:
            flag = self.extend(T, q)
            if flag is not ExtendState.advanced:
                return flag

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim

        return node_new

    def is_node_same(self, node_a, node_b):
        dist, theta = self.get_distance_and_angle(node_a, node_b)
        if dist<1e-6:
            return True

        return False

    def generate_random_node(self, sample_goal, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return sample_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(list(reversed(path1)) + path2)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node

    rrt_conn = RRTConnect(x_start, x_goal, 0.8, 0.05, 5000)
    path = rrt_conn.planning()

    rrt_conn.plotting.animation_connect(rrt_conn.T1, rrt_conn.T2, path, "RRT_CONNECT")


if __name__ == '__main__':
    main()
