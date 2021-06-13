import numpy as np
import logging

### logger system ###
logger_name = __name__
logger = logging.Logger(logger_name)
file_handler = logging.FileHandler(f'{__name__}.log','w')
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)
### end logger system ###

class GridMap(object):
    def __init__(self, range_min, range_max, range_num, min_angle, max_angle, angle_increment, grid_size=0.05, wall_thickness = 0.35, map_size=[50,50], init_xy=[0,0], log_odds=np.log(4), log_odds_max = 10):
        """
        Args:
            range_min, range_max, range_num, min_angle, max_angle, angle_increment: lidar parameters
            grid_size, wall_thickness, map_size, init_xy: map parameters
        """
        super().__init__()
        
        ### init scan settings
        self.range_min_ = range_min
        self.range_max_ = range_max
        self.min_angle_ = min_angle
        self.max_angle_ = max_angle
        self.angle_increment_ = angle_increment
        self.range_num_ = range_num

        log_data = {
            'range_min': self.range_min_,
            'range_max': self.range_max_,
            'range_num': self.range_num_,
            'min_angle': self.min_angle_,
            'max_angle': self.max_angle_,
            'angle_increment': self.angle_increment_

        }
        logger.info(f'\n{log_data}')
        ### end scan settings

        ### init map settings
        self.grid_size_ = grid_size
        self.wall_thickness_ = wall_thickness
        self.map_origin_ = np.array(init_xy, dtype=float)
        self.map_size_ = np.array(map_size)
        self.grid_map_ = np.zeros([0,0])   # map of log odds
        self.expandMap(map_size[0]/2,map_size[1]/2)
        self.expandMap(-map_size[0]/2,-map_size[1]/2)
        self.log_odds = log_odds
        self.log_odds_max = log_odds_max
        log_data = {
            'grid_size': self.grid_size_,
            'wall_thickness': self.wall_thickness_,
            'map_origin': self.map_origin_,
            'map_size': self.map_size_,
            'log_odds': self.log_odds,
            'log_odds_max': self.log_odds_max
        }
        logger.info(f'\n{log_data}')
        ### end map settings

        # discretize rays
        rays_x_, rays_y_, rays_len_ = self.discretizeRay()
        self.rays_x_ = rays_x_
        self.rays_y_ = rays_y_
        self.rays_len_ = rays_len_

    ### public interface
    def update(self, x_t, scan_t):
        """Update the grid map with the scan and robot pose
        Args:
            x_t: [x, y, yaw]
            scan_t : ndarray: shape=(range_num_,)
        """
        x_t = np.array(x_t)
        scan_t = np.array(scan_t)
        assert x_t.shape==(3,) and scan_t.shape==(self.range_num_,)
        
        x, y, yaw = x_t
        # update map size
        if (x+self.range_max_)+5>self.map_origin_[0]+self.map_size_[0]:
            self.expandMap(x+self.range_max_+5-(self.map_origin_[0]+self.map_size_[0]), 0)
        if (x-self.range_max_)-5<self.map_origin_[0]:
            self.expandMap((x-self.range_max_)-5-self.map_origin_[0], 0)
        if (y+self.range_max_)+5>self.map_origin_[1]+self.map_size_[1]:
            self.expandMap(0, (y+self.range_max_)+5-(self.map_origin_[1]+self.map_size_[1])) 
        if (y-self.range_max_)-5<self.map_origin_[1]:
            self.expandMap(0, (y-self.range_max_)-5-self.map_origin_[1])
        logger.info(f'map origin {self.map_origin_}')
        logger.info(f'map size {self.map_size_}')


        # world_T_robot
        cyaw = np.cos(yaw)
        syaw = np.sin(yaw)
        T = np.array([[cyaw,-syaw ,x],
                      [syaw,cyaw  ,y],
                      [0   ,0     ,1]])

        # update log odd map
        p_free = [] # list of 2d points
        p_occ = []
        for scan_ind, scan in enumerate(scan_t):
            if self.range_min_<scan<self.range_max_:
                p_inds = np.where(self.rays_len_[scan_ind] < scan)[0]
                p_xs = self.rays_x_[scan_ind][p_inds]
                p_ys = self.rays_y_[scan_ind][p_inds]
                for p_x, p_y in zip(p_xs, p_ys):
                    p_free.append([p_x, p_y])

                p_inds = np.where(np.logical_and(self.rays_len_[scan_ind] > scan, self.rays_len_[scan_ind] < scan+self.wall_thickness_))[0]
                p_xs = self.rays_x_[scan_ind][p_inds]
                p_ys = self.rays_y_[scan_ind][p_inds]
                for p_x, p_y in zip(p_xs, p_ys):
                    p_occ.append([p_x, p_y])
            else:
                p_inds = np.where(self.rays_len_[scan_ind] < self.range_max_)[0]
                p_xs = self.rays_x_[scan_ind][p_inds]
                p_ys = self.rays_y_[scan_ind][p_inds]
                for p_x, p_y in zip(p_xs, p_ys):
                    p_free.append([p_x, p_y])
        def rigidTransform(bP, aTb):
            p_homo = np.zeros([3,len(bP)])
            p_homo[0,:] = bP[:,0]
            p_homo[1,:] = bP[:,1]
            p_homo[2,:] = 1
            aP = aTb @ p_homo
            return aP[:2].T
        if len(p_free)>0:
            p_free = rigidTransform(np.array(p_free), T)
            for p in p_free:
                x_ind, y_ind = self.pos2ind(p[0], p[1])
                self.grid_map_[x_ind, y_ind] -= self.log_odds
                self.grid_map_[x_ind, y_ind] = max(self.grid_map_[x_ind, y_ind], -self.log_odds_max)
        if len(p_occ)>0:
            p_occ = rigidTransform(np.array(p_occ), T)
            for p in p_occ:
                x_ind, y_ind = self.pos2ind(p[0], p[1])
                self.grid_map_[x_ind, y_ind] += self.log_odds
                self.grid_map_[x_ind, y_ind] = min(self.grid_map_[x_ind, y_ind], self.log_odds_max)
        

    def getProbabilityMap(self):
        """Return the label of probability map. Each grid represents P(m=1|x,z) occupancy probability.
        """
        gamma = 1-np.ones_like(self.grid_map_)/(np.exp(self.grid_map_)+1)
        return gamma

    ### end public interface
        
    def expandMap(self, x, y):
        """expand the grid map with extra x, y (meters)
        Args:
            x, y: meters 
        """
        x_grid_num = abs(int(x/self.grid_size_))
        y_grid_num = abs(int(y/self.grid_size_))
        if x>=0:
            self.grid_map_ = np.concatenate([self.grid_map_, np.zeros([x_grid_num,self.grid_map_.shape[1]])],axis=0)
        else:
            self.grid_map_ = np.concatenate([np.zeros([x_grid_num,self.grid_map_.shape[1]]), self.grid_map_],axis=0)
            self.map_origin_[0] -= x_grid_num*self.grid_size_
        if y>=0:
            self.grid_map_ = np.concatenate([self.grid_map_, np.zeros([self.grid_map_.shape[0],y_grid_num])],axis=1)
        else:
            self.grid_map_ = np.concatenate([np.zeros([self.grid_map_.shape[0],y_grid_num]), self.grid_map_],axis=1)
            self.map_origin_[1] -= y_grid_num*self.grid_size_
        self.map_size_ =(self.grid_map_.shape[0]*self.grid_size_, self.grid_map_.shape[1]*self.grid_size_)

    
    def discretizeRay(self):
        """Discretize the rays to points
        """
        # generate a square
        s = np.linspace(-self.range_max_-self.wall_thickness_-1, self.range_max_+self.wall_thickness_+1, num=int((self.range_max_*2)/self.grid_size_)+1)
        x_mesh, y_mesh = np.meshgrid(s,s)
        r_mesh = np.hypot(x_mesh,y_mesh)
        angle_mesh = np.rint(np.mod((np.arctan2(y_mesh, x_mesh)-self.min_angle_),2*np.pi)/self.angle_increment_)
        rays_x = []
        rays_y = []
        rays_len = []
        for angle_ind in range(self.range_num_):
            inds = np.argwhere(angle_mesh==angle_ind)
            rays_x.append(x_mesh[inds[:,0],inds[:,1]])
            rays_y.append(y_mesh[inds[:,0],inds[:,1]])
            rays_len.append(r_mesh[inds[:,0],inds[:,1]])
        return rays_x, rays_y, rays_len

    def pos2ind(self, x, y):
        """Return the index of a position
        """
        x_ind = int(round((x-self.map_origin_[0])/self.grid_size_))
        y_ind = int(round((y-self.map_origin_[1])/self.grid_size_))
        return x_ind, y_ind

import matplotlib.pyplot as plt
def main():
    m = GridMap(1.0,20,180,-90/180*np.pi, 90/180*np.pi, 1/180*np.pi)
    pos = np.array([0,0,0.0])
    intel_laser = np.loadtxt('dataset_intel/intel_LASER_.txt')
    intel_odom = np.loadtxt('dataset_intel/intel_ODO.txt')
    traj = []
    for i in range(len(intel_laser)):
        scan = intel_laser[i]
        odom = intel_odom[i]
        cyaw = np.cos(pos[-1])
        syaw = np.sin(pos[-1])

        pos[:2] += np.array([[cyaw, -syaw],
                           [syaw,  cyaw]]) @ odom[:2]
        pos[2] += odom[2]
        m.update(pos, scan)
        traj.append(pos.copy())
        if (i)%100==0:
            plt.cla()
            plt.imshow((m.grid_map_.T)[::-1], cmap='gray', extent=[m.map_origin_[0], m.map_origin_[0]+m.map_size_[0], m.map_origin_[1], m.map_origin_[1]+m.map_size_[1]])
            t = np.array(traj)
            plt.scatter(t[:,0], t[:,1], s=0.2)
            # plot scan
            scan_pts = np.zeros([len(scan), 2])
            
            for scan_ind in range(len(scan)):
                scan_pts[scan_ind,0] = pos[0]+scan[scan_ind]*np.cos(pos[-1]+scan_ind*m.angle_increment_+m.min_angle_)
                scan_pts[scan_ind,1] = pos[1]+scan[scan_ind]*np.sin(pos[-1]+scan_ind*m.angle_increment_+m.min_angle_)
            plt.scatter(scan_pts[:,0], scan_pts[:,1], c='r', s=0.1)
            plt.savefig(f'{i}_step.png')
if __name__ == '__main__':
    main()