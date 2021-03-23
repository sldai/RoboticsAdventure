"""
DYNAMIC_RRT_2D
@author: huiming zhou

"""

import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

import env, plotting, utils

class Flag(Enum):
    VALID = 1
    INVALID = 2

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.flag = Flag.VALID


class Edge:
    def __init__(self, n_p, n_c):
        self.parent = n_p
        self.child = n_c
        self.flag = Flag.VALID


class DynamicRRT:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, waypoint_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.waypoint_sample_rate = waypoint_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.vertex_old = []
        self.vertex_new = []
        self.edges = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()
        self.fig, self.ax = plt.subplots()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_add = [0, 0, 0] # [x,y,r]

        self.path = []
        self.waypoint = []

    def grow_rrt(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate, self.waypoint_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.edges.append(Edge(node_near, node_new))
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)
                if dist <= self.step_len:
                    self.new_state(node_new, self.s_goal)

                    self.path = self.extract_path(node_new)
                    self.waypoint = self.extract_waypoint(node_new)
                    return self.path
        print("No path is found")
        return None

    def regrow_rrt(self):
        self.old_path = self.path
        print("Trimming the tree ...")
        self.trim_rrt()
        print("replanning path ...")
        path = self.grow_rrt()
        return path

    def trim_rrt(self):
        waypoint = []
        for node in self.waypoint:
            if node.flag == Flag.VALID:
                waypoint.append(node)
            elif node.flag == Flag.INVALID:
                break
        self.waypoint = waypoint
        self.old_waypoint = waypoint
        for i in range(1, len(self.vertex)):
            node = self.vertex[i]
            node_p = node.parent
            if node_p.flag == Flag.INVALID:
                node.flag = Flag.INVALID

        self.vertex = [node for node in self.vertex if node.flag == Flag.VALID]
        self.vertex_old = copy.deepcopy(self.vertex)
        self.edges = [Edge(node.parent, node) for node in self.vertex[1:len(self.vertex)]]


    def is_collision_obs_add(self, start, end):
        delta = self.utils.delta
        obs_add = self.obs_add

        if math.hypot(start.x - obs_add[0], start.y - obs_add[1]) <= obs_add[2] + delta:
            return True

        if math.hypot(end.x - obs_add[0], end.y - obs_add[1]) <= obs_add[2] + delta:
            return True

        o, d = self.utils.get_ray(start, end)
        if self.utils.is_intersect_circle(o, d, [obs_add[0], obs_add[1]], obs_add[2]):
            return True

        return False

    def invalidate_nodes(self):
        # find affected edges 
        for edge in self.edges:
            if self.is_collision_obs_add(edge.parent, edge.child):
                edge.child.flag = Flag.INVALID

    def is_path_invalid(self):
        for node in self.waypoint:
            if node.flag == Flag.INVALID:
                return True


    def generate_random_node(self, goal_sample_rate, waypoint_sample_rate):
        delta = self.utils.delta
        p = np.random.random()

        if p < goal_sample_rate:
            return self.s_goal
        elif goal_sample_rate < p < goal_sample_rate + waypoint_sample_rate:
            if len(self.waypoint)>0:
                return self.waypoint[np.random.randint(0, len(self.waypoint))]
            else:
                return self.s_goal
        else:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

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

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    def extract_waypoint(self, node_end):
        waypoint = [node_end]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            waypoint.append(node_now)

        return waypoint

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > 50 or y < 0 or y > 30:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Add circle obstacle at: x =", x, ",", "y =", y)
            self.obs_add = [x, y, 2]
            self.obs_circle.append([x, y, 2])
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle)
            self.plotting.obs_circle.append([x, y, 2])
            self.invalidate_nodes()

            if self.is_path_invalid():
                self.regrow_rrt()

                print("len_vertex: ", len(self.vertex))
                print("len_vertex_old: ", len(self.vertex_old))
                print("len_vertex_new: ", len(self.vertex)-len(self.vertex_old))

                print("waypoint number", len(self.old_waypoint))

                plt.cla()
                self.plotting.plot_grid("Dynamic_RRT")
                self.plotting.plot_visited(self.vertex_old, animation=False, c="-g")
                # self.plotting.plot_path(self.old_path, c='-b')
                self.plotting.plot_visited(self.old_waypoint, animation=False, c='-b')
                self.plotting.plot_visited(self.vertex[len(self.vertex_old):], animation=True, c="-m")
                
                self.plotting.plot_path(self.path)
            else:
                print("Trimming Invalid Nodes ...")
                self.trim_rrt()
                plt.cla()
                self.plotting.plot_grid("Dynamic_RRT")
                self.plotting.plot_visited(self.vertex, animation=False, c="-g")
                self.plotting.plot_path(self.path)

            self.fig.canvas.draw_idle()




def main():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node

    drrt = DynamicRRT(x_start, x_goal, 0.5, 0.1, 0.6, 10000)
    drrt.grow_rrt()
    drrt.plotting.animation(drrt.vertex, drrt.path, "Dynamic RRT", animation=True, show=False)
    drrt.fig.canvas.mpl_connect('button_press_event', drrt.on_press)
    

    plt.show()


if __name__ == '__main__':
    main()
