"""
D* 
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

import plotting, env, utils, my_queue


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.child = []

class DStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type, res=0.5):
        self.res = res
        self.s_start = self.pos2ind(s_start)
        self.s_goal = self.pos2ind(s_goal)
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env
        self.utils = utils.Utils()
        self.u_set = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        self.OPEN = my_queue.HeapDict()  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = {}  # recorded parent
        self.g = {}  # cost to come

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

    def compute_shortest_path(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        while len(self.OPEN)>0 and self.OPEN.top()[-1]<self.g[self.s_goal]:
            s = self.OPEN.get()
            self.CLOSED.append(s)

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    self.OPEN.put(s_n, self.f_value(s_n))
            
            if s == self.s_goal:  # stop condition
                break
        # transform the index set to node set
        path = self.extract_path(self.PARENT)
        path = self.trans_path(path)
        node_list = self.trans_node_list(self.PARENT.keys())
        return path, node_list

    def main(self):
        self.PARENT[self.s_start] = None
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        self.OPEN.put(self.s_start, self.f_value(self.s_start))
        path, node_list = self.compute_shortest_path()
        self.plot = plotting.Plotting(self.s_start, self.s_goal)
        self.plot.animation(node_list, path, "A*", show=False)
        self.fig = plt.gcf()
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()

    def trim(self):
        # extract tree from visited vertices
        T = {}
        for v in self.PARENT.keys():
            node = Node(self.ind2pos(v))
            node.parent = self.PARENT[v]
            T[v] = node

        for u in self.PARENT.keys():
            if u != self.s_start:
                v = self.PARENT[u]
                T[v].child.append(u)
        
        # trim the branch if the edge is affected
        # use BFS to delete the branch
        q = [T[self.s_start]]
        while len(q)>0:
            v = q.pop(0)
            child_new = []
            for u in v.child:
                # if (v,u) is affected
                # delete the branch of u
                if not self.is_collision_obs_add(v,Node(self.ind2pos(u))):
                    child_new.append(u)
                    q.append(T[u])
            v.child = child_new
    
        # recover the A* information
        CLOSED = []
        PARENT = {}
        g = {}
        OPEN = my_queue.HeapDict()

        q = [T[self.s_start]]
        while len(q)>0:
            v = q.pop(0)
            for u in v.child:
                q.append(T[u])
            v_ind = self.pos2ind((v.x,v.y))
            PARENT[v_ind]=self.PARENT[v_ind]
            g[v_ind]=self.g[v_ind]
            if self.OPEN.find(v_ind):
                # condition: if the open node is still 
                # in the tree, put it in the open list
                OPEN.put(v_ind,self.f_value(v_ind))
            else:
                # condition: if the node has affected edges
                # we need to search from the node to see whether a
                # better solution exists
                x, y = self.ind2pos(v_ind)
                node_src = Node((x,y))
                for u in self.u_set:
                    dx = u[0]*self.res
                    dy = u[1]*self.res
                    node_dst = Node((x+dx,y+dy))
                    if self.is_collision_obs_add(node_src,node_dst):
                        OPEN.put(v_ind,self.f_value(v_ind))
                        break
        self.PARENT = PARENT
        self.g = g
        self.OPEN = OPEN
        self.CLOSED = CLOSED
    
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

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > 50 or y < 0 or y > 30:
            print("Please choose right area!")
        else:
            # # move the goal
            # self.s_goal = self.PARENT[self.s_goal]
            
            x, y = int(x), int(y)
            if (not hasattr(self, 'obs_add_rm')) or self.obs_add_rm is False:
                print("Add circle obstacle at: x =", x, ",", "y =", y)
                self.obs_add = [x, y, 2]
                self.utils.obs_circle.append(self.obs_add)
                self.plot.obs_circle.append([x, y, 2])
                self.obs_add_rm = True
            else:
                self.obs_add = [x, y, 2]
                self.utils.obs_circle.pop(-1)
                self.plot.obs_circle.pop(-1)
                self.obs_add_rm = False
            print("trimming the tree")
            print(f'Before trimming, tree node {len(self.PARENT)}, open node {len(self.OPEN)}')
            self.trim()
            print(f'After trimming, tree node {len(self.PARENT)}, open node {len(self.OPEN)}')  
            self.node_old = [k for k in self.PARENT.keys()]
            if self.s_goal not in self.PARENT.keys():
                self.g[self.s_goal] = math.inf
            
            print("replanning ...")
            path, node_list = self.compute_shortest_path()
            old, new = self.split_old_new_nodes()
            plt.cla()
            self.plot.plot_visited(new, False, c='-b')
            self.plot.animation(old, path, "D*", show=False)
            
            self.fig.canvas.draw_idle()
    def split_old_new_nodes(self):
        all_nodes = self.trans_node_list(self.PARENT.keys())
        old = []
        new = []
        for node in all_nodes:
            if self.pos2ind((node.x,node.y)) in self.node_old:
                old.append(node)
            else:
                new.append(node)
        return old, new

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
            if self.PARENT[k]:
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
        return self.g[s] + self.heuristic(s)

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

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node
        p_s = self.ind2pos(s)
        p_g = self.ind2pos(goal)
        if heuristic_type == "manhattan":
            return abs(p_g[0] - p_s[0]) + abs(p_g[1] - p_s[1])
        else:
            return math.hypot(p_g[0] - p_s[0], p_g[1] - p_s[1])


def main():
    s_start = (2, 2)
    s_goal = (49, 24)

    planner = DStar(s_start, s_goal, "euclidean", res=1.0)
    planner.main()



if __name__ == '__main__':
    main()
