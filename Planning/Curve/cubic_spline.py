#! /usr/bin/python
# -*- coding: utf-8 -*-
u"""
Cubic Spline library on python

author Atsushi Sakai

usage: see test codes as below

license: MIT
"""
import math
import numpy as np
import bisect


class Spline:
    u"""
    Cubic Spline class
    Given m+1 control points, fit m piecewise polynomials
    $s_i(x)=a_i(x-x_i)^3+b_i(x-x_i)^2+c_i(x-x_i)+d_i$
    
    See also: http://macs.citadel.edu/chenm/343.dir/09.dir/lect3_4.pdf
    """
    def __init__(self, x, y):
        assert len(x) == len(y), 'x, y shape do not match'
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.a, self.b, self.c, self.d = [], [], [], []

        self.nx = len(x)  # dimension of s
        h = np.diff(x)

        self.a = [y for y in self.y]

        self.c = self.__calc_c(h)
        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)
    
    def __calc_c(self, h):
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[-1, -2] = 0.0
        A[-1, -1] = 1.0

        B = np.zeros(self.nx)
        y = self.a
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (y[i + 2] - y[i + 1]) / \
                h[i + 1] - 3.0 * (y[i + 1] - y[i]) / h[i]

        c = np.linalg.solve(A, B)
        return c

    def calc(self, t):
        u"""
        Calc position

        if t is outside of the input s, return None

        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        u"""
        Calc first derivative

        if t is outside of the input s, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        u"""
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        u"""
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1


class Spline2D:
    u"""
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [1.0
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        u"""
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        u"""
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)**(3/2)
    
        return k

    def calc_yaw(self, s):
        u"""
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw

    def get_path(self, step_size=0.1):
        s = np.arange(0, self.s[-1], step_size)
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = self.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(self.calc_yaw(i_s))
            rk.append(self.calc_curvature(i_s)) 
        return rx, ry, ryaw, rk       

def test_spline2d():
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    x = [0, 10, 20, 30, 40,30,20,10,0]
    y = [0,10,0,-10,0,10,0,-10,0]

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], 0.1)

    rx, ry, ryaw, rk = sp.get_path()
    # for i_s in s:
    #     ix, iy = sp.calc_position(i_s)
    #     rx.append(ix)
    #     ry.append(iy)
    #     ryaw.append(sp.calc_yaw(i_s))
    #     rk.append(sp.calc_curvature(i_s))

    flg, ax = plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title('cubic spline 2D')
    plt.legend()

    flg, ax = plt.subplots(1)
    plt.plot(s, [math.degrees(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    flg, ax = plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()


def test_spline():
    print("Spline test")
    import matplotlib.pyplot as plt
    x = [-2, 2.0, 3.5, 5.5, 6.0, 8.0]
    y = [0, 2.7, -0.5, 0.5, 3.0, 4.0]

    spline = Spline(x, y)
    rx = np.arange(np.min(x), np.max(x), 0.01)
    ry = [spline.calc(i) for i in rx]

    plt.plot(x, y, "xb")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('cubic spline')
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    # test_spline()
    test_spline2d()
