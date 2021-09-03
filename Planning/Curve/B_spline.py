import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from typing import List


class BSpline(object):
    def __init__(self, knots: np.ndarray, control_points: np.ndarray, degree: int = 3):
        """See also https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node17.html#fig:rep_cvx_hull_bspl
        """
        n = len(control_points)-1
        k = degree+1
        assert len(knots) == n+1+k
        self.n = n  # 0->n control points
        self.k = k  # k order, k-1 degree
        self.T = np.array(knots)
        self.p = np.array(control_points)

    def calcBasisFunction(self, t):
        """Calculate all basis function values in time t
        """
        N = np.zeros([self.n+1, self.k+1])  # basis function
        # order 1 is constant
        for i in range(self.n+1):
            N[i, 1] = 1 if self.T[i] <= t < self.T[i+1] else 0

        # order k->k+1, see https://en.wikipedia.org/wiki/B-spline
        for k in range(1, self.k):
            for i in range(self.n+1):
                w_i_k = (t-self.T[i])/(self.T[i+k]-self.T[i]
                                       ) if self.T[i+k] != self.T[i] else 0
                w_i1_k = (t-self.T[i+1])/(self.T[i+1+k]-self.T[i+1]
                                          ) if self.T[i+1+k] != self.T[i+1] else 0
                if i < self.n:
                    N[i, k+1] = w_i_k*N[i, k]+(1-w_i1_k)*N[i+1, k]
                else:
                    N[i, k+1] = w_i_k*N[i, k]
        return N

    def DeBoor(self, t):
        # constrain t in [u_{k}, u_{n+1}]
        ub = min(max(self.T[self.k-1], t), self.T[-self.k])

        # determine which [ui,ui+1] lay in
        k = self.k-1
        while True:
            if self.T[k + 1] >= ub:
                break
            k += 1

        # deBoor's alg
        c = self.p
        p = self.k-1
        x = ub
        t = self.T
        d = [c[j + k - p] for j in range(0, p + 1)]

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        return d[p]

    @staticmethod
    def parameterizeToBspline(point_set: List[List]):
        """Fit BSpline curve to pass point_set, see also: https://hal.archives-ouvertes.fr/hal-03017566/document
        Args:
            ts: time step between points
            point_set: points to be fitted
            start_end_derivative: [start_vel, end_vel, start_acc, end_acc]

        Returns:
            ctrl_pts: control points of the uniform BSpline

        """
        K = len(point_set)

        prow = np.array([1, 4, 1], dtype=float)

        # K pos, 2 vel, 2 acc constraints, K+2 control points
        A = np.zeros([K, K + 2])
        for i in range(K):
            A[i:i+1, i:i+3] = (1 / 6.0) * prow

        # write b
        bx = np.zeros(K)
        by = np.zeros(K)

        for i in range(K):
            bx[i] = point_set[i][0]
            by[i] = point_set[i][1]

        # solve Ax = b
        px = np.linalg.lstsq(A, bx, rcond=None)[0]
        py = np.linalg.lstsq(A, by, rcond=None)[0]

        # convert to control pts
        ctrl_pts = np.zeros([K+2, 2])
        ctrl_pts[:, 0] = px
        ctrl_pts[:, 1] = py
        return ctrl_pts


def splineEval():
    control_points = np.array([(3, 1), (2.5, 4), (0, 1), (-2.5, 4),
                               (-3, 0), (-2.5, -4)])
    clamped_knots = [0, 0, 0, 0, 1/3, 2/3, 1, 1, 1, 1]
    spl = BSpline(clamped_knots, control_points)
    clamped_bspline = np.array([spl.DeBoor(t) for t in np.linspace(0, 1)]).T

    unclamped_knots = [-0.9, -0.6, -0.3, 0, 1/3, 2/3, 1, 1.3, 1.6, 1.9]
    spl = BSpline(unclamped_knots, control_points)
    unclamped_bspline = np.array([spl.DeBoor(t) for t in np.linspace(0, 1)]).T

    nonuniform_knots = [0, 0, 0, 0, 1/4, 4/5, 1, 1, 1, 1]
    spl = BSpline(nonuniform_knots, control_points)
    nonuniform_knots = np.array([spl.DeBoor(t) for t in np.linspace(0, 1)]).T

    plt.plot(clamped_bspline[0], clamped_bspline[1], "--",
             linewidth=2.0, label='Clamped B-spline curve')
    plt.plot(unclamped_bspline[0], unclamped_bspline[1],
             "-.", linewidth=2.0, label='Unclamped B-spline curve')
    plt.plot(nonuniform_knots[0], nonuniform_knots[1],
             linewidth=2.0, label='Nonuniform B-spline curve')
    plt.plot(control_points[:, 0], control_points[:, 1], 'k--',
             label='Control polygon', marker='o', markerfacecolor='red')
    plt.legend()
    plt.title("BSpline Evaluate")
    plt.show()


def splineInterp():
    points = np.array([(3, 1), (2.5, 4), (0, 1), (-2.5, 4),
                       (-3, 0), (-2.5, -4)])
    ctrl_pts = BSpline.parameterizeToBspline(points)
    K = len(points)

    knots = list(range(-3, K+3))
    spl = BSpline(knots, ctrl_pts)
    interp_result = np.array([spl.DeBoor(t)
                              for t in np.linspace(0, K-1, num=100)]).T
    plt.plot(points[:, 0], points[:, 1], 'sk',
             label='Passing points', markerfacecolor='darkorange')
    plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], '--ok',
             label='Control polygon', markerfacecolor='red')
    plt.plot(interp_result[0], interp_result[1],
             linewidth=2, c="cyan", label="Interpolated curve")
    plt.legend()
    plt.title("BSpline Interpolate")
    plt.show()


if __name__ == "__main__":
    splineEval()
    splineInterp()
