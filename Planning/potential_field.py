"""
Potential Field based path planner
author: Atsushi Sakai (@Atsushi_twi)
Modified by Sldai 
Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 3

show_animation = True


def calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy):
    minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    # astar search
    dist_map, path = Astar_search(gx, gy, sx, sy, ox, oy, 1, reso, minx, miny, maxx, maxy)
    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            # ug = calc_attractive_potential(x, y, gx, gy)
            ug = calc_attractive_potential_expanded(ix, iy, dist_map)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny, path

class AstarNode:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_ind = 0
        self.y_ind = 0
        self.cost = np.inf
        self.parent = None

def Astar_search(sx, sy, gx, gy, ox, oy, rr, reso, minx, miny, maxx, maxy):
    obs_set = set()
    def posToInd(x,y):
        x_ind = round((x-minx)/reso)
        y_ind = round((y-miny)/reso)
        return x_ind, y_ind

    def indToPos(x_ind, y_ind):
        x = minx+x_ind*reso
        y = miny+y_ind*reso
        return x, y
    for _ox, _oy in zip(ox,oy):
        for dx in np.arange(-rr, rr+reso, reso):
            for dy in np.arange(-rr, rr+reso, reso):
                if np.hypot(dx,dy)<=rr:
                    x_ind, y_ind = posToInd(_ox+dx, _oy+dy)
                    obs_set.add((x_ind, y_ind))
    
    node_map = {}
    open_set = PriorityQueue()
    closed_set = []
    start_node = AstarNode()
    start_node.x = sx
    start_node.y = sy
    start_node.x_ind, start_node.y_ind = posToInd(sx, sy)
    start_node.parent = None
    start_node.cost = 0.0
    node_map[(start_node.x_ind, start_node.y_ind)] = start_node
    def addNodeToOpen(n):
        d = n.cost
        h = np.hypot(gx-n.x, gy-n.y)
        open_set.put((d+h, d, n.x_ind, n.y_ind, n))
    addNodeToOpen(start_node)
    while not open_set.empty():
        s = open_set.get()[-1]
        closed_set.append(s)
        if np.hypot(gx-s.x, gy-s.y)<reso:
            break
        for u in get_motion_model():
            n_ind = (s.x_ind+u[0], s.y_ind+u[1])
            if n_ind in obs_set:
                continue
            if n_ind not in node_map.keys():
                node_map[n_ind] = AstarNode()
                node_map[n_ind].x, node_map[n_ind].y = indToPos(n_ind[0], n_ind[1])
                node_map[n_ind].x_ind, node_map[n_ind].y_ind = n_ind
                node_map[n_ind].parent = None
                node_map[n_ind].cost = np.inf
            n = node_map[n_ind]

            if s.cost+np.hypot(n.x-s.x, n.y-s.y)< n.cost:
                n.parent = s
                n.cost = s.cost+np.hypot(n.x-s.x, n.y-s.y)
                addNodeToOpen(n)
    
    # return the map{pos: cost}
    dist_map = {(n.x_ind, n.y_ind):n.cost for n in node_map.values()}

    # extract course
    path = []
    while s is not None:
        path.append((s.x_ind,s.y_ind))
        s=s.parent
    path.reverse()
    return dist_map, path



def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)
def calc_attractive_potential_expanded(x_ind, y_ind, dist_map):
    ind = (x_ind, y_ind)
    if ind not in dist_map.keys():
        return 1000
    else:
        return 0.5 * KP * dist_map[(x_ind, y_ind)]

def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False
from scipy import ndimage
def calc_direction(x_direction, y_direction, x, y, interpolate=True):
    """calculate the direction for x, y (float)
    """
    if not interpolate:
        r_x = round(x)
        r_y = round(y)
        directon = -np.array([x_direction[r_x, r_y], y_direction[r_x, r_y]])
    else:
        l_x = int(np.floor(x))
        u_x = int(np.floor(x+1))
        l_y = int(np.floor(y))
        u_y = int(np.floor(y+1))
        q11 = -np.array([x_direction[l_x, l_y], y_direction[l_x, l_y]])
        q12 = -np.array([x_direction[l_x, u_y], y_direction[l_x, u_y]])
        q21 = -np.array([x_direction[u_x, l_y], y_direction[u_x, l_y]])
        q22 = -np.array([x_direction[u_x, u_y], y_direction[u_x, u_y]])

        directon = (q11 * (u_x - x) * (u_y - y) +
            q21 * (x - l_x) * (u_y - y) +
            q12 * (u_x - x) * (y - l_y) +
            q22 * (x - l_x) * (y - l_y)
           ) / ((u_x - l_x) * (u_y - l_y) + 0.0)
    return directon

def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):

    # calc potential field
    pmap, minx, miny, path = calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy)
    path = np.array(path)
    x_direction = ndimage.sobel(pmap,axis=0,mode="constant")
    y_direction = ndimage.sobel(pmap,axis=1,mode="constant")
    norm = np.hypot(x_direction, y_direction)+1e-6
    x_direction = x_direction/norm
    y_direction = y_direction/norm
    

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)

    if show_animation:
        draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")
        print(ox,oy)
        plt.plot(np.round((np.array(ox)-minx)/reso), np.round((np.array(oy)-miny)/reso), "o", c="cyan")
        plt.plot(path[:,0], path[:,1],c="orange")


    rx, ry = [sx], [sy]
    motion = get_motion_model()
    previous_ids = deque()
    while d >= reso:
        # minp = float("inf")
        # minix, miniy = -1, -1
        # for i, _ in enumerate(motion):
        #     inx = int(ix + motion[i][0])
        #     iny = int(iy + motion[i][1])
        #     if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
        #         p = float("inf")  # outside area
        #         print("outside potential!")
        #     else:
        #         p = pmap[inx][iny]
        #     if minp > p:
        #         minp = p
        #         minix = inx
        #         miniy = iny
        direction = calc_direction(x_direction, y_direction, ix, iy)
        minix = ix+direction[0]
        miniy = iy+direction[1]

        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)

        if (oscillations_detection(previous_ids, ix, iy)):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break

        if show_animation:
            plt.plot(ix, iy, ".r")
            plt.pause(0.01)

    print("Goal!!")

    return rx, ry


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def test_astar():
    sx = 0.0  # start x position [m]
    sy = 10.0  # start y positon [m]
    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]
    reso = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    ox = [15.0, 5.0, 20.0, 20.0]  # obstacle x position list [m]
    oy = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]
    minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    dist_map = Astar_search(gx, gy, sx, sy, ox, oy, robot_radius, reso, minx, miny, maxx, maxy)
    x_arr = [x for x, y in dist_map.keys()]
    y_arr = [y for x, y in dist_map.keys()]
    dist_arr = [v for v in dist_map.values()]
    plt.scatter(x_arr, y_arr, cmap=dist_arr)
    print(dist_map.keys())

def main():
    print("potential_field_planning start")

    sx = 0.0  # start x position [m]
    sy = 10.0  # start y positon [m]
    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    ox = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
    oy = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]
    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    _, _ = potential_field_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")