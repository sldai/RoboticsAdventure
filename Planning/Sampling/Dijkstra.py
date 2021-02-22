"""
Dijkstra
@author: huiming zhou
Modified by Shilong Dai
"""

import os
import sys
import math
import heapq

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

import plotting, env, utils, my_queue


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None

class Dijkstra:
    """Dijkstra set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, res=0.5):
        self.res = res
        self.s_start = self.pos2ind(s_start)
        self.s_goal = self.pos2ind(s_goal)
        self.Env = env.Env()  # class Env
        self.utils = utils.Utils()
        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]

        self.OPEN = my_queue.QueuePrior()  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def ind2pos(self, ind):
        """Convert vertex index to vertex position
        """
        x = ind[0] * self.res
        y = ind[1] * self.res
        return (x,y)
    
    def pos2ind(self, pos):
        """Convert vertex position to vertex index
        """
        x_ind = int(pos[0]/self.res)
        y_ind = int(pos[1]/self.res)
        return (x_ind,y_ind)

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        self.OPEN.put(self.s_start, self.f_value(self.s_start))

        while self.OPEN:
            s = self.OPEN.get()
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    self.OPEN.put(s_n, self.f_value(s_n))

        # transform the index set to node set
        path = self.extract_path(self.PARENT)
        path = self.trans_path(path)
        node_list = self.trans_node_list(self.CLOSED)
        return path, node_list
    
    def trans_path(self, path_ind):
        """Transform the path with inds to path with pos
        """
        path_pos = []
        for i in path_ind:
            path_pos.append(self.ind2pos(i))
        return path_pos

    def trans_node_list(self, CLOSED):
        """Transform the CLOSED index set to a node list
        """
        node_dict = dict()
        for k,v in self.PARENT.items():
            node_dict[k] = Node(self.ind2pos(k))
        for k,v in node_dict.items():
            node_dict[k].parent = node_dict[self.PARENT[k]]
    
        node_list = []
        for k in CLOSED:
            node_list.append(node_dict[k])
        return node_list

    def get_neighbor(self, s):
        """Get neighbors of s 
        Args:
            s (node): a visited node
        
        Returns:
            n_list (node list): neighbor nodes of s
        """
        x, y = self.ind2pos(s)
        n_list = []
        for u in self.u_set:
            dx = u[0]*self.res
            dy = u[1]*self.res
            ind_tmp = self.pos2ind((x+dx,y+dy))
            if not self.is_collision(s,ind_tmp):
                n_list.append(ind_tmp)
        return n_list

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        p_s = self.ind2pos(s_start)
        p_g = self.ind2pos(s_goal)
        return math.hypot(p_g[0] - p_s[0], p_g[1] - p_s[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start index
        :param s_end: end index
        :return: True: is collision / False: not collision
        """
        start_node = Node(self.ind2pos(s_start))
        end_node = Node(self.ind2pos(s_end))
        return self.utils.is_collision(start_node,end_node)


    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """
        return self.g[s]

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)


def main():
    s_start = (2, 2)
    s_goal = (49, 24)

    astar = Dijkstra(s_start, s_goal, res=1.0)
    plot = plotting.Plotting(s_start, s_goal)
    path, visited = astar.searching()
    plot.animation(visited, path, "Dijkstra*")  # animation



if __name__ == '__main__':
    main()
