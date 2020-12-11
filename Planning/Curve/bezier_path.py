"""
bezier path: 
ref:
https://en.wikipedia.org/wiki/B%C3%A9zier_curve
https://github.com/zhm-real/MotionPlanning
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import draw

class BezierCurve(object):
    """Bezier polynomial curve is defined as 
    the sum of Bernstein basis polynomials
    1 = (t+(1-t))^{n} = \sum_{v=0}^{n}b_{v,n}(t)
    b_{v,n}(t) = \binom{v}{n} t^v (1-t)^{n-v}
    """
    def __init__(self, control_points):
        """
        Args:
            control_points (array_like): [p0,p1,...,pn]
        """
        super().__init__()
        self.control_points = np.array(control_points)
        self.degree = len(control_points)-1
        n = self.degree
        self.w = self.bezier_derivatives_control_points(self.control_points, n)

    @staticmethod
    def Bernstein_func(n, v, t):
        return comb(n,v) * np.power(t,v) * np.power(1-t,n-v)
    
    @staticmethod
    def bezier_derivatives_control_points(control_points, n_derivatives):
        """
        A derivative of a bezier curve is a bezier curve.
        See https://pomax.github.io/bezierinfo/#derivatives
        for detailed explanations
        """
        w = {0: control_points}
        for i in range(n_derivatives):
            n = len(w[i])-1
            w[i + 1] = np.array([n * (w[i][j + 1] - w[i][j])
                                for j in range(n)])
        return w

    def calc_deriv(self, t, order=0):
        """Calculate the nth order derivative at t
        Args:
            t (float, array): t \in [0,1]
        
        Returns:
            point (1d vector, 2d array)
        """
        control_points = self.w[order]
        n = len(control_points)-1
        t = np.array(t)
        if len(t.shape)>0:
            point = np.zeros([len(t), control_points.shape[1]])
            for v in range(n+1):
                weights = self.Bernstein_func(n, v, t)    
                point += weights[:,None] * control_points[v]
        else: # scalar
            point = np.zeros_like(control_points[0])
            for v in range(n+1):
                weights = self.Bernstein_func(n, v, t)    
                point += weights * control_points[v]
        return point

    def calc_path(self, num=100):
        """A wrapper to get bezier path
        Args:
            num (int): number of way points

        Returns:
            path (array): way points
        """
        path = self.calc_deriv(np.linspace(0,1,num), 0)
        return path


def curvature(dx, dy, ddx, ddy):
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)


def main():
    sx, sy, syaw = 10.0, 1.0, np.deg2rad(180.0)
    gx, gy, gyaw = 0.0, -3.0, np.deg2rad(-45.0)

    offset = 3.0
    dist = np.hypot(sx - gx, sy - gy) / offset
    control_points = np.array(
        [[sx, sy],
         [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
         [gx - dist * np.cos(gyaw), gy - dist * np.sin(gyaw)],
         [gx, gy]])

    bz = BezierCurve(control_points)
    path = bz.calc_path(100)
    t = 0.8
    point = bz.calc_deriv(t,0)
    pdt = bz.calc_deriv(t,1)
    pddt = bz.calc_deriv(t,2)
    # Radius of curv
    radius = 1 / curvature(pdt[0], pdt[1], pddt[0], pddt[1])
    # Normalize derivative
    pdt /= np.linalg.norm(pdt, 2)
    tangent = np.array([point, point + pdt])
    normal = np.array([point, point + [- pdt[1], pdt[0]]])
    curvature_center = point + np.array([- pdt[1], pdt[0]]) * radius

    assert path[0,0] == sx and path[0,1] == sy, "path is invalid"
    assert path[-1,0] == gx and path[-1,1] == gy, "path is invalid"

    fig, ax = plt.subplots()
    yaw = np.linspace(-np.pi, np.pi)
    circle = curvature_center+radius*np.array([np.cos(yaw), np.sin(yaw)]).T
    ax.plot(control_points.T[0], control_points.T[1],
            '--o', label="Control Points")

    plt.plot(path[:,0],path[:,1],label='Bezier Curve')
    ax.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
    ax.plot(normal[:, 0], normal[:, 1], label="Normal")
    ax.plot(circle[:,0],circle[:,1],c='cyan',label='Curvature Circle')
    draw.Arrow(sx, sy, syaw, 1, "darkorange")
    draw.Arrow(gx, gy, gyaw, 1, "darkorange")
    plt.grid(True)
    plt.title("Bezier Path")
    ax.axis("equal")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

