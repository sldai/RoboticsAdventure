"""Python implementation of Fast Planner kinodynamic trajectory generation
ref: https://github.com/HKUST-Aerial-Robotics/Fast-Planner
"""

from queue import PriorityQueue
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class NonUniformBspline(object):
    """An implementation of non-uniform B-spline with different dimensions.
    It also represents uniform B-spline which is a special case of non-uniform
    """

    def __init__(self, points: np.ndarray, degree: int, interval: float):
        self.control_points = None  # numpy array dim (n, 3)
        self.p = None  # p degree
        self.n = None  # n control points
        self.m = None  # m+1 knots, m = n+1+p
        self.u = None  # knot vector
        self.interval = None
        self.limit_vel = None  # physical limits
        self.limit_acc = None
        self.limit_ratio = None  # and time adjustment ratio

        self.setUniformBspline(points, degree, interval)

    def setUniformBspline(self, control_points: np.ndarray, degree: int, interval: float):
        self.control_points = np.array(control_points)
        assert self.control_points.shape[0] >= 4 and self.control_points.shape[
            1] == 2, f"control_points.shape: {self.control_points.shape}"
        self.p = degree
        self.interval = interval

        self.n = self.control_points.shape[0] - 1
        self.m = self.n + self.p + 1

        self.u = np.zeros(self.m+1)
        for i in range(len(self.u)):
            if i <= self.p:
                self.u[i] = (-self.p+i)*self.interval
            elif self.p < i <= self.m-self.p:
                self.u[i] = self.u[i-1]+self.interval
            elif i > self.m-self.p:
                self.u[i] = self.u[i-1]+self.interval

    def setKnot(self, knot: np.ndarray):
        self.u = np.array(knot)
        assert len(self.u) == self.m+1, f"u length: {len(self.u)}"

    def getKnot(self):
        return np.array(self.u)

    def getTimeSpan(self):
        um = self.u[self.p]
        um_p = self.u[self.m-self.p]
        return um, um_p

    def getControlPoint(self):
        return np.array(self.control_points)

    def getHeadTailPts(self):
        um, um_p = self.getTimeSpan()
        head = self.evaluateDeBoor(um)
        tail = self.evaluateDeBoor(um_p)
        return head, tail

    def evaluateDeBoor(self, u):
        um, um_p = self.getTimeSpan()
        ub = np.clip(u, um, um_p)

        # determine which [ui,ui+1] lay in
        k = self.p
        while True:
            if self.u[k+1] >= ub:
                break
            k += 1

        # deBoor
        d = []
        for i in range(self.p+1):
            d.append(self.control_points[k - self.p + i])

        for r in range(1, self.p+1):
            for i in range(self.p, r-1, -1):
                alpha = (ub-self.u[i+k-self.p]) / \
                    (self.u[i+1+k-r]-self.u[i+k-self.p])
                d[i] = (1-alpha)*d[i-1]+alpha*d[i]
        return d[self.p]

    def getDerivativeControlPoints(self):
        """The derivative of a b-spline is also a b-spline, its degree become p-1
        control point Q_i = p*(P_{i+1}-P_i)/(u_{i+p+1}-u_{i+1})
        see also: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
        """
        ctp = np.zeros([self.control_points.shape[0] -
                        1, self.control_points.shape[1]])
        P = self.control_points
        for i in range(ctp.shape[0]):
            ctp[i] = self.p*(P[i+1]-P[i])/(self.u[i+self.p+1]-self.u[i+1])
        return ctp

    def getDerivative(self):
        ctp = self.getDerivativeControlPoints()
        derivative = NonUniformBspline(ctp, self.p - 1, self.interval)
        derivative.setKnot(self.u[1:-1])
        return derivative

    def getInterval(self):
        return self.interval

    def setPhysicalLimits(self, vel: float, acc: float):
        self.limit_vel = vel
        self.limit_acc = acc
        self.limit_ratio = 1.1

    def reallocateTime(self):
        fea = True
        P = self.control_points
        dim = P.shape[1]
        max_vel, max_acc = None, None

        # check vel feasibility and insert points
        for i in range(P.shape[0]-1):
            vel = self.p * (P[i + 1] - P[i]) / \
                (self.u[i + self.p + 1] - self.u[i + 1])
            if np.any(np.abs(vel) > self.limit_vel+1e-4):
                fea = False
                max_vel = np.max(np.abs(vel))
                ratio = min(self.limit_ratio, max_vel / self.limit_vel)
                time_ori = self.u[i + self.p + 1] - self.u[i + 1]
                time_new = ratio * time_ori
                delta_t = time_new - time_ori
                t_inc = delta_t / self.p
                for j in range(i+2, i + self.p + 2):
                    self.u[j] += (j-i-1) * t_inc
                for j in range(j+self.p+2, len(self.u)):
                    self.u[j] += delta_t

        # acc
        for i in range(P.shape[0] - 2):
            acc = self.p * (self.p - 1) * ((P[i + 2] - P[i + 1]) / (self.u[i + self.p + 2] - self.u[i + 2]) - (
                P[i + 1] - P[i]) / (self.u[i + self.p + 1] - self.u[i + 1])) / (self.u[i + self.p + 1] - self.u[i + 2])
            if np.any(np.abs(acc) > self.limit_acc+1e-4):
                fea = False
                max_acc = np.max(np.abs(acc))
                ratio = min(self.limit_ratio, np.sqrt(
                    max_acc / self.limit_acc))
                time_ori = self.u[i + self.p + 1] - self.u[i + 2]
                time_new = ratio * time_ori
                delta_t = time_new - time_ori
                t_inc = delta_t / (self.p - 1)
                for j in range(i + 3, i + self.p + 2):
                    self.u[j] += (j - i - 2) * t_inc
                for j in range(i + self.p + 2, len(self.u)):
                    self.u[j] += delta_t
        return fea

    @staticmethod
    def parameterizeToBspline(ts: float, point_set: np.ndarray, start_end_derivative: np.ndarray):
        """Given waypoints and derivative constraints, fit a uniform cubic Bspline, return the control point
        Args:
            ts: interval
            point_set: 2D waypoints
            start_end_derivative: [start_vel, end_vel, start_acc, end_acc]
        """
        assert ts > 0, f"ts: {ts}"
        assert len(point_set) >= 2, f"len(point_set): {len(point_set)}"
        assert len(
            start_end_derivative) >= 4, f"len(start_end_derivative): {len(start_end_derivative)}"
        K = len(point_set)

        # construct A
        prow = np.array([1, 4, 1], dtype=float)
        vrow = np.array([-1, 0, 1], dtype=float)
        arow = np.array([1, -2, 1], dtype=float)

        A = np.zeros((K + 4, K + 2))
        for i in range(K):
            A[i, i:i+3] = (1 / 6.0) * prow
        A[K, 0:3] = (1 / 2.0 / ts) * vrow
        A[K+1, K - 1:K+2] = (1 / 2.0 / ts) * vrow

        A[K + 2, 0:3] = (1 / ts / ts) * arow
        A[K + 3, K - 1:K+2] = (1 / ts / ts) * arow

        # construct b
        bx = np.zeros(K + 4)
        by = np.zeros(K + 4)
        for i in range(K):
            bx[i] = point_set[i][0]
            by[i] = point_set[i][1]

        for i in range(4):
            bx[K + i] = start_end_derivative[i][0]
            by[K + i] = start_end_derivative[i][1]

        # solve Ax = b
        W = np.eye(len(A))  # least square weight
        px = np.linalg.lstsq(W@A, W@bx, rcond=None)[0]
        py = np.linalg.lstsq(W@A, W@by, rcond=None)[0]

        # convert to control pts
        ctrl_pts = np.zeros([K + 2, 2])
        ctrl_pts[:, 0] = px
        ctrl_pts[:, 1] = py
        return ctrl_pts


class BsplineOptimizer(object):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env
        self.control_points = None
        self.bspline_interval = None
        self.end_pt = None
        self.dim = 2
        self.guide_pts = None
        self.waypoints = None
        self.waypt_ind = None
        self.cost_function = None

        self.order = 3
        self.lambda1 = 1.0  # smoothness
        self.lambda2 = 5.0  # distance weight
        self.lambda3 = 0.00001  # feasibility weight
        self.lambda4 = 0.01  # end point weight
        self.lambda5 = 0.0  # guide cost weight
        self.lambda6 = 0.0  # visibility cost weight
        self.lambda7 = 100.0  # waypoint cost weight

        self.dist0 = 0.4
        self.max_vel = 3.0
        self.max_acc = 2.0
        self.visib_min = None
        self.max_iteration_num = 300
        self.max_iteration_time = None

        self.variable_num = None
        self.iter_num = None
        self.best_variable = None
        self.min_cost = None
        self.block_pts = None

        self.SMOOTHNESS = (1 << 0)
        self.DISTANCE = (1 << 1)
        self.FEASIBILITY = (1 << 2)
        self.ENDPOINT = (1 << 3)
        self.GUIDE = (1 << 4)
        self.WAYPOINTS = (1 << 6)

    def setEnv(self, env):
        self.env = env

    def setControlPoints(self, points):
        self.control_points = np.array(points)
        self.dim = self.control_points.shape[1]
        assert self.dim == 2, f"self.dim: {self.dim==2}"
        assert len(
            self.control_points) > 6, f"len(self.control_points)={len(self.control_points)}"

    def setInterval(self, ts):
        self.bspline_interval = ts

    def setTerminatedCond(self, max_iteration_num, max_iteration_time):
        self.max_iteration_num = max_iteration_num
        self.max_iteration_time = max_iteration_time

    def setCostFunction(self, cost_code):
        self.cost_function = cost_code
        cost_str = ""
        if self.cost_function & self.SMOOTHNESS:
            cost_str += "smooth |"
        if self.cost_function & self.DISTANCE:
            cost_str += " distance |"
        if self.cost_function & self.FEASIBILITY:
            cost_str += " FEASIBILITY |"
        if self.cost_function & self.ENDPOINT:
            cost_str += " ENDPOINT |"
        if self.cost_function & self.GUIDE:
            cost_str += " GUIDE |"
        if self.cost_function & self.WAYPOINTS:
            cost_str += " WAYPOINTS |"
        print(cost_str)

    def setGuidePath(self, guide_pts):
        self.guide_pts = np.array(guide_pts)
        assert guide_pts.shape[1] == self.dim

    def setWaypoints(self, waypoints, waypt_ind):
        self.waypoints = np.array(waypoints)
        self.waypt_ind = [ind for ind in waypt_ind]
        assert self.waypoints.shape[1] == self.dim

    def BsplineOptimizeTraj(self, points, ts, cost_function, max_iteration_num, max_iteration_time):
        self.setControlPoints(points)
        self.setInterval(ts)
        self.setCostFunction(cost_function)
        self.setTerminatedCond(max_iteration_num, max_iteration_time)
        self.optimize()
        return self.control_points

    def optimize(self):
        # init solver
        self.iter_num = 0
        self.min_cost = np.inf
        pt_num = len(self.control_points)
        self.variable_num = self.dim * (pt_num - 2 * self.order)
        x0 = np.zeros(self.variable_num)
        for i in range(self.order, self.variable_num//self.dim+self.order):
            for j in range(self.dim):
                x0[self.dim*(i-self.order)+j] = self.control_points[i, j]
        bound_upper = x0+10.0
        bound_lower = x0-10.0
        bound = [(lower, upper)
                 for lower, upper in zip(bound_lower, bound_upper)]

        result = minimize(self.costFunction, x0, jac=True,
                          bounds=bound, options={"maxiter":self.max_iteration_num})
        self.best_variable = result.x
        print(result)
        for i in range(self.order, self.variable_num//self.dim+self.order):
            for j in range(self.dim):
                self.control_points[i,
                                    j] = self.best_variable[self.dim * (i - self.order) + j]
        return self.control_points

    def costFunction(self, x):
        """Callback of scipy minimize function
        Args:
            x: shape (variable_num,), current variable

        Returns:
            f_combine: current cost
            grad: current gradient
        """
        cost, grad = self.combineCost(x)
        self.iter_num += 1

        # save the min cost result
        if cost < self.min_cost:
            self.min_cost = cost
            self.best_variable = x
        return cost, grad

    def combineCost(self, x):
        """
        Args:
            x: shape (variable_num,) current variable

        Returns:
            f_combine: current cost
            grad: shape (variable_num,), current gradient
        """
        # convert the NLopt format vector to control points.
        control_points = np.zeros_like(self.control_points)
        control_points[:self.order, :] = self.control_points[:self.order, :]
        for i in range(self.variable_num // self.dim):
            for j in range(self.dim):
                control_points[i + self.order, j] = x[self.dim * i + j]
        control_points[self.order + self.variable_num //
                       self.dim:] = self.control_points[self.order + self.variable_num // self.dim:]

        f_combine = 0.0
        grad = np.zeros(self.variable_num)
        f_smoothness = f_distance = f_feasibility = f_endpoint = f_guide = f_waypoints = 0.0
        if self.cost_function & self.SMOOTHNESS:
            f_smoothness, g_smoothness = self.calcSmoothnessCost(
                control_points)
            f_combine += self.lambda1 * f_smoothness
            for i in range(self.variable_num//self.dim):
                for j in range(self.dim):
                    grad[self.dim*i+j] += self.lambda1 * \
                        g_smoothness[i + self.order, j]
        if self.cost_function & self.DISTANCE:
            f_distance, g_distance = self.calcDistanceCost(control_points)
            f_combine += self.lambda2 * f_distance
            for i in range(self.variable_num//self.dim):
                for j in range(self.dim):
                    grad[self.dim*i+j] += self.lambda2 * \
                        g_distance[i + self.order, j]
        if self.cost_function & self.FEASIBILITY:
            f_feasibility, g_feasibility = self.calcFeasibilityCost(
                control_points)
            f_combine += self.lambda3 * f_feasibility
            for i in range(self.variable_num//self.dim):
                for j in range(self.dim):
                    grad[self.dim*i+j] += self.lambda3 * \
                        g_feasibility[i + self.order, j]
        if self.cost_function & self.ENDPOINT:
            f_endpoint, g_endpoint = self.calcEndpointCost(control_points)
            f_combine += self.lambda4 * f_endpoint
            for i in range(self.variable_num//self.dim):
                for j in range(self.dim):
                    grad[self.dim*i+j] += self.lambda4 * \
                        g_endpoint[i + self.order, j]
        if self.cost_function & self.GUIDE:
            f_guide, g_guide = self.calcGuideCost(control_points)
            f_combine += self.lambda5 * f_guide
            for i in range(self.variable_num//self.dim):
                for j in range(self.dim):
                    grad[self.dim*i+j] += self.lambda5 * \
                        g_guide[i + self.order, j]
        if self.cost_function & self.WAYPOINTS:
            f_waypoints, g_waypoints = self.calcWaypointsCost(control_points)
            f_combine += self.lambda7 * f_waypoints
            for i in range(self.variable_num//self.dim):
                for j in range(self.dim):
                    grad[self.dim*i+j] += self.lambda7 * \
                        g_waypoints[i + self.order, j]
        return f_combine, grad

    def calcEbandCost(self, x):
        f = 0.0
        g = np.zeros_like(x)

        for i in range(len(x)-2):
            # evaluate tensor
            tensor = -x[i]+2*x[i+1]-x[i+2]
            f += tensor@tensor
            temp_t = 2*tensor
            # tensor gradient
            g[i] += -temp_t
            g[i+1] += 2*temp_t
            g[i+2] += -temp_t
        return f, g

    def calcSmoothnessCost(self, x):
        """
        Args:
            x: shape==control_points.shape, variable

        Returns:
            f: cost
            g: shape==x.shape, gradient
        """
        f = 0.0
        g = np.zeros_like(x)

        for i in range(len(x)-self.order):
            # evaluate jerk
            jerk = x[i + 3] - 3 * x[i + 2] + 3 * x[i + 1] - x[i]
            f += jerk @ jerk
            temp_j = 2.0 * jerk
            # jerk gradient
            g[i + 0] += -temp_j
            g[i + 1] += 3.0 * temp_j
            g[i + 2] += -3.0 * temp_j
            g[i + 3] += temp_j
        return f, g

    def calcDistanceCost(self, x):
        f = 0.0
        g = np.zeros_like(x)

        end_idx = len(x) - self.order

        for i in range(self.order, end_idx):
            dist, dist_grad = self.env.evaluateDistGrad(x[i])
            if np.linalg.norm(dist_grad) > 1e-4:
                dist_grad /= np.linalg.norm(dist_grad)

            if dist < self.dist0:
                f += (dist - self.dist0)**2
                g[i] += 2.0 * (dist - self.dist0) * dist_grad
        return f, g

    def calcFeasibilityCost(self, x):
        f = 0.0
        g = np.zeros_like(x)

        # abbreviation
        vm2 = self.max_vel**2
        am2 = self.max_acc**2

        ts = self.bspline_interval
        ts_inv2 = 1 / ts / ts
        ts_inv4 = ts_inv2 * ts_inv2

        # velocity feasibility
        for i in range(len(x)-1):
            vi = x[i + 1] - x[i]
            for j in range(self.dim):
                vd = vi[j] * vi[j] * ts_inv2 - vm2
                if vd > 0.0:
                    f += vd ** 2

                    temp_v = 4.0 * vd * ts_inv2
                    g[i + 0, j] += -temp_v * vi[j]
                    g[i + 1, j] += temp_v * vi[j]

        # acceleration feasibility
        for i in range(len(x)-2):
            ai = x[i + 2] - 2 * x[i + 1] + x[i]

            for j in range(self.dim):
                ad = ai[j] * ai[j] * ts_inv4 - am2
                if ad > 0.0:
                    f += ad ** 2

                    temp_a = 4.0 * ad * ts_inv4
                    g[i + 0, j] += temp_a * ai[j]
                    g[i + 1, j] += -2 * temp_a * ai[j]
                    g[i + 2, j] += temp_a * ai[j]
        return f, g

    def calcEndpointCost(self, x):
        f = 0.0
        g = np.zeros_like(x)
        return f, g

    def calcGuideCost(self, x):
        f = 0.0
        g = np.zeros_like(x)
        return f, g

    def calcWaypointsCost(self, x):
        f = 0.0
        g = np.zeros_like(x)
        return f, g


class ComplexEnv(object):
    def __init__(self) -> None:
        super().__init__()
        self.center = np.array([0, 0])
        self.length = np.array((8, 6))
        self.origin = self.center-self.length/2
        self.reso = 0.1
        self.size = np.array(self.length/self.reso, dtype=int)

        self.occ_map = np.zeros(self.size)
        self.esdf_map = np.zeros(self.size)

        self.circles = []
        self.rectangles = [(-2, -1, 4, 2)]

        self.genOcc()
        self.distanceTransform()

    def posToInd(self, pos: np.ndarray):
        ind = np.round((np.array(pos)-self.origin)/self.reso)
        ind = np.array(ind, dtype=int)
        return ind

    def indToPos(self, ind: np.ndarray):
        pos = np.array(ind, dtype=float)*self.reso+self.origin
        return pos

    def boundIndex(self, ind):
        return np.clip(ind, np.zeros(2, dtype=int), self.size-1, dtype=int)

    def isInside(self, pos):
        ind = self.posToInd(pos)
        return np.all(ind >= 0) and np.all(ind < self.size)

    def addCircle(self, o, r):
        self.circles.append((o[0], o[1], r))

    def addRectange(self, o, w, h):
        self.rectangles.append((o[0], o[1], w, h))

    def genOcc(self):
        for ox, oy, r in self.circles:
            lb = self.posToInd((ox-r, oy-r))
            rt = self.posToInd((ox+r, oy+r))
            for x_ind in range(lb[0], rt[0]):
                for y_ind in range(lb[1], rt[1]):
                    x, y = self.indToPos((x_ind, y_ind))
                    if not self.isInside((x, y)):
                        continue
                    if np.hypot(x-ox, y-oy) <= r:
                        self.occ_map[x_ind, y_ind] = 1

        for ox, oy, w, h in self.rectangles:
            lb = self.posToInd((ox, oy))
            rt = self.posToInd((ox+w, oy+h))
            for x_ind in range(lb[0], rt[0]):
                for y_ind in range(lb[1], rt[1]):
                    x, y = self.indToPos((x_ind, y_ind))
                    if not self.isInside((x, y)):
                        continue
                    self.occ_map[x_ind, y_ind] = 1

    def distanceTransform(self):
        """Get esdf from occmap, see also https://www.cs.cornell.edu/~dph/papers/dt.pdf
        """
        tmp_buffer2 = tmp_buffer1 = np.zeros_like(self.esdf_map)

        def f_set_val(buf, x, y, val):
            buf[x, y] = val

        # calculate positive distance
        for x in range(self.occ_map.shape[0]):
            self.FillESDF(
                lambda y: 0 if self.occ_map[x, y] == 1 else 1000000,
                lambda y, val: f_set_val(tmp_buffer1, x, y, val),
                0, self.occ_map.shape[1]-1, 1
            )

        for y in range(self.occ_map.shape[1]):
            self.FillESDF(
                lambda x: tmp_buffer1[x, y],
                lambda x, val: f_set_val(tmp_buffer2, x, y, val),
                0, self.occ_map.shape[0]-1, 0
            )
        dist_pos = np.sqrt(tmp_buffer2)*self.reso

        # calculate negative distance
        tmp_buffer2 = tmp_buffer1 = np.zeros_like(self.esdf_map)

        # calculate positive distance
        for x in range(self.occ_map.shape[0]):
            self.FillESDF(
                lambda y: 0 if self.occ_map[x, y] == 0 else 1000000,
                lambda y, val: f_set_val(tmp_buffer1, x, y, val),
                0, self.occ_map.shape[1]-1, 1
            )

        for y in range(self.occ_map.shape[1]):
            self.FillESDF(
                lambda x: tmp_buffer1[x, y],
                lambda x, val: f_set_val(tmp_buffer2, x, y, val),
                0, self.occ_map.shape[0]-1, 0
            )
        dist_neg = np.sqrt(tmp_buffer2)*self.reso
        self.esdf_map = dist_pos - dist_neg

    def FillESDF(self, f_get_val, f_set_val, start, end, dim):
        v = np.zeros(self.esdf_map.shape[dim], dtype=int)
        z = np.zeros(self.esdf_map.shape[dim]+1)

        k = start
        v[start] = start
        z[start] = -np.inf
        z[start+1] = np.inf

        for q in range(start+1, end+1):
            k += 1

            s = -np.inf
            while s <= z[k]:
                k -= 1
                # if q==243 and k==242:
                #     print(f_get_val(q), f_get_val(v[k]), q, v[k])
                s = ((f_get_val(q) + q*q) -
                     (f_get_val(v[k]) + v[k]*v[k])) / (2*q - 2*v[k])

            k += 1

            v[k] = q
            z[k] = s
            z[k + 1] = np.inf

        k = start
        for q in range(start, end+1):
            while z[k + 1] < q:
                k += 1
            val = (q - v[k]) * (q - v[k]) + f_get_val(v[k])
            f_set_val(q, val)

    def evaluateDistGrad(self, pos):
        pos = np.array(pos)
        pos_m = self.indToPos(self.posToInd(pos - self.reso / 2))
        diff = (pos - pos_m) / self.reso
        values = np.zeros([2, 2])
        for x in range(2):
            for y in range(2):
                values[x, y] = self.getDistance(
                    pos_m+np.array([x*self.reso, y*self.reso]))
        dist = (values[0, 0]*(1-diff[0])*(1-diff[1]) +
                values[0, 1]*(1-diff[0])*(diff[1]) +
                values[1, 0]*(diff[0])*(1-diff[1]) +
                values[1, 1]*(diff[0])*(diff[1]))
        grad = np.zeros(2)
        grad[0] = (values[1, 0]-values[0, 0])*(1-diff[1]) + \
            (values[1, 1]-values[0, 1])*(diff[1])
        grad[1] = (values[0, 1]-values[0, 0])*(1-diff[0]) + \
            (values[1, 1]-values[1, 0])*(diff[0])
        grad = grad/self.reso
        return dist, grad

    def getDistance(self, pos):
        ind = self.posToInd(pos)
        ind = self.boundIndex(ind)
        return self.esdf_map[ind[0], ind[1]]

    def isValid(self, pos):
        ind = self.posToInd(pos)
        ind = self.boundIndex(ind)
        return self.occ_map[ind[0], ind[1]] == 0  # check occ

    def getBbox(self):
        """Return the bounding box
        Returns:
            origin: 
            length:
        """
        return np.array(self.origin), np.array(self.length)


def plotEnv(env: ComplexEnv):
    fig = plt.gcf()
    ax = plt.gca()
    for ox, oy, r in env.circles:
        ax.add_patch(patches.Circle(
            (ox, oy), r, edgecolor='black', facecolor='gray', fill=True))
    for ox, oy, w, h in env.rectangles:
        ax.add_patch(patches.Rectangle((ox, oy), w, h,
                                       edgecolor='black', facecolor='gray', fill=True))


class AstarNode:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_ind = 0
        self.y_ind = 0
        self.cost = np.inf
        self.parent = None


def AstarSearch(reso, start, end, env: ComplexEnv):
    """Astar search path
    Args:
        reso: float
        start: [x,y]
        end: [x,y]
        env: ComplexEnv
    Returns:
        path: [[x,y]]
    """

    start = np.array(start)
    end = np.array(end)
    center = (start+end)/2

    def posToInd(x, y):
        x_ind = round((x-center[0])/reso)
        y_ind = round((y-center[1])/reso)
        return x_ind, y_ind

    def indToPos(x_ind, y_ind):
        x = center[0]+x_ind*reso
        y = center[1]+y_ind*reso
        return x, y
    node_map = {}
    open_set = PriorityQueue()
    start_node = AstarNode()
    start_node.x_ind, start_node.y_ind = posToInd(start[0], start[1])
    start_node.x, start_node.y = indToPos(start_node.x_ind, start_node.y_ind)
    start_node.parent = None
    start_node.cost = 0.0

    def addNodeToOpen(node):
        g = node.cost
        h = np.hypot(end[0]-node.x, end[1]-node.y)
        open_set.put((g+h, g, node.x_ind, node.y_ind, node))
    addNodeToOpen(start_node)
    node_map[(start_node.x_ind, start_node.y_ind)] = start_node
    cnt = 400
    while not open_set.empty() and cnt > 0:
        cur_node = open_set.get()[-1]

        if np.hypot(cur_node.x-end[0], cur_node.y-end[1]) < reso:
            break
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                pro_x = cur_node.x + dx*reso
                pro_y = cur_node.y + dy*reso
                if not env.isValid((pro_x, pro_y)):
                    continue
                pro_x_ind, pro_y_ind = posToInd(pro_x, pro_y)
                assert not (
                    pro_x_ind == cur_node.x_ind and pro_y_ind == cur_node.y_ind)
                if (pro_x_ind, pro_y_ind) not in node_map.keys():
                    pro_node = AstarNode()
                    pro_node.x = pro_x
                    pro_node.y = pro_y
                    pro_node.x_ind = pro_x_ind
                    pro_node.y_ind = pro_y_ind
                    pro_node.parent = None
                    pro_node.cost = np.inf
                    node_map[(pro_x_ind, pro_y_ind)] = pro_node
                pro_node = node_map[(pro_x_ind, pro_y_ind)]
                if cur_node.cost + np.hypot(dx, dy)*reso < pro_node.cost:
                    pro_node.parent = cur_node
                    pro_node.cost = cur_node.cost + np.hypot(dx, dy)*reso
                    addNodeToOpen(pro_node)

    # extract course
    cur_node = cur_node
    path = []
    while cur_node != None:
        path.append([cur_node.x, cur_node.y])
        cur_node = cur_node.parent
    path.reverse()
    return path


def calcVel(x, t):
    assert len(x) == len(t)
    x = np.array(x)
    dx = np.diff(x, axis=0)
    dt = np.diff(t)
    v = np.array([_dx/_dt for _dx, _dt in zip(dx, dt)])
    return v


def calcAcc(x, t):
    assert len(x) == len(t)
    v = calcVel(x, t)
    a = calcVel(v, t[:-1])
    return a

start = (-2.5, 0.0)
end = (3.0, 0.0)
start_v = (0.0,0.0)
start_a = (0.0,0.0)
end_v = (0.0,0.0)
end_a = (0.0,0.0)
reso = 0.2
ts = 0.1

figwidth = 6
# A star search initial path
complex_env = ComplexEnv()
path = AstarSearch(reso, start, end, complex_env)
path = np.array(path)

origin, length = complex_env.getBbox()
fig = plt.figure(figsize=(figwidth, figwidth/length[0]*length[1]))
plt.axis([origin[0], origin[0]+length[0], origin[1], origin[1]+length[1]])
plotEnv(complex_env)
plt.plot(path[:,0],path[:,1])
plt.title("Initial path")

# convert path to B-spline trajectory
ctrl_pts = NonUniformBspline.parameterizeToBspline(ts, path, np.zeros([4,2]))
bs_raw = NonUniformBspline(ctrl_pts, 3, ts)
t_min, t_max = bs_raw.getTimeSpan()
traj_t = np.linspace(t_min,t_max,num=100)
traj_raw = np.array([bs_raw.evaluateDeBoor(t) for t in traj_t])

origin, length = complex_env.getBbox()
fig = plt.figure(figsize=(figwidth, figwidth/length[0]*length[1]))
plt.axis(np.array([origin[0], origin[0]+length[0], origin[1], origin[1]+length[1]]))
plotEnv(complex_env)
plt.plot(ctrl_pts[:,0],ctrl_pts[:,1],"--ok",linewidth=2)
plt.plot(traj_raw[:,0],traj_raw[:,1],linewidth=2)
plt.title("Initial Control Points")

# B-spline control points optimization
opt = BsplineOptimizer(complex_env)
ctrl_pts_opt = opt.BsplineOptimizeTraj(ctrl_pts, ts, opt.SMOOTHNESS|opt.DISTANCE|opt.FEASIBILITY, 500, 500)
bspline_opt = NonUniformBspline(ctrl_pts_opt, 3, ts)

# uncomment below to reallocate time to ensure feasibility
# bspline_opt.setPhysicalLimits(3,2)
# while True:
#     if bspline_opt.reallocateTime():
#         break
# print(bspline_opt.getTimeSpan())

t_min, t_max = bspline_opt.getTimeSpan()
traj_t = np.linspace(t_min, t_max, num=100)
traj_opt = np.array([bspline_opt.evaluateDeBoor(t) for t in traj_t])

origin, length = complex_env.getBbox()
fig = plt.figure(figsize=(figwidth, figwidth/length[0]*length[1]))
plt.axis(np.array([origin[0], origin[0]+length[0], origin[1], origin[1]+length[1]]))
plotEnv(complex_env)
plt.plot(ctrl_pts_opt[:,0],ctrl_pts_opt[:,1], '--ok')
plt.plot(traj_opt[:,0],traj_opt[:,1], c='purple')
plt.title("Optimized Bspline")

plt.figure(figsize=(figwidth, figwidth/length[0]*length[1]))
traj_opt_vel = calcVel(traj_opt, traj_t)
traj_opt_acc = calcAcc(traj_opt, traj_t)
plt.plot(traj_t[:-1], traj_opt_vel[:,0], "-", linewidth=2, label="vel_x")
plt.plot(traj_t[:-1], traj_opt_vel[:,1], "-", linewidth=2, label="vel_y")
plt.plot(traj_t[:-2], traj_opt_acc[:,0], "-", linewidth=2, label="acc_x")
plt.plot(traj_t[:-2], traj_opt_acc[:,1], "-", linewidth=2, label="acc_y")
plt.legend()
plt.title("Optimized Bspline Vel/Acc")
plt.show()
