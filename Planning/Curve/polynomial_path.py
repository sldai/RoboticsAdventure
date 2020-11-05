
import math
import numpy as np
import matplotlib.pyplot as plt

import draw


class CubicPolynomial:
    def __init__(self, x0, v0, x1, v1, T):
        self.a0 = x0
        self.a1 = v0

        A = np.array(
            [[T**2, T**3],
             [2*T, 3*T**2]]
        )
        b = np.array(
            [[x1-self.a0-self.a1*T],
             [v1-self.a1]]
        )
        X = np.linalg.solve(A, b)
        self.a2 = X[0]
        self.a3 = X[1]

    def calc_xt(self, t):
        return self.a0+self.a1*t+self.a2*t**2+self.a3*t**3

    def calc_dxt(self, t):
        return self.a1+self.a2*2*t+self.a3*3*t**2

    def calc_ddxt(self, t):
        return self.a2*2+self.a3*6*t


class QuinticPolynomial:
    def __init__(self, x0, v0, a0, x1, v1, a1, T):
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0/2.0

        A = np.array(
            [[T**3, T**4, T**5],
             [3*T**2, 4*T**3, 5*T**4],
             [6*T, 12*T**2, 20*T**3]]
        )
        b = np.array(
            [[x1-self.a0-self.a1*T-self.a2*T**2],
             [v1-self.a1-2*self.a2*T],
             [a1-2*self.a2]]
        )

        X = np.linalg.solve(A, b)
        self.a3 = X[0]
        self.a4 = X[1]
        self.a5 = X[2]

    def calc_xt(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
            self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_dxt(self, t):
        dxt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return dxt

    def calc_ddxt(self, t):
        ddxt = 2 * self.a2 + 6 * self.a3 * t + 12 * \
            self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return ddxt

    def calc_dddxt(self, t):
        dddxt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return dddxt


class CubicPolynomial2D:
    def __init__(self, x0, vx0, x1, vx1, y0, vy0, y1, vy1, T):
        self.x_cubic = CubicPolynomial(x0, vx0, x1, vx1, T)
        self.y_cubic = CubicPolynomial(y0, vy0, y1, vy1, T)

    def calc_position(self, t):
        r"""
        calc position
        """
        x = self.x_cubic.calc_xt(t)
        y = self.y_cubic.calc_xt(t)

        return x, y

    def calc_curvature(self, t):
        r"""
        calc curvature
        """
        dx = self.x_cubic.calc_dxt(t)
        ddx = self.x_cubic.calc_ddxt(t)
        dy = self.y_cubic.calc_dxt(t)
        ddy = self.y_cubic.calc_ddxt(t)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)**(3/2)
        return k

    def calc_speed(self, t):
        r"""
        calc speed
        """
        dx = self.x_cubic.calc_dxt(t)
        dy = self.y_cubic.calc_dxt(t)
        v = math.hypot(dy, dx)
        return v

    def calc_yaw(self, t):
        r"""
        calc yaw
        """
        dx = self.x_cubic.calc_dxt(t)
        dy = self.y_cubic.calc_dxt(t)
        yaw = math.atan2(dy, dx)
        return yaw

    def calc_acc(self, t):
        r"""
        calc acceleration
        """
        ddx = self.x_cubic.calc_ddxt(t)
        ddy = self.y_cubic.calc_ddxt(t)
        return ddx, ddy


class QuinticPolynomial2D:
    def __init__(self, x0, vx0, ax0, x1, vx1, ax1, y0, vy0, ay0, y1, vy1, ay1, T):
        self.x_cubic = QuinticPolynomial(x0, vx0, ax0, x1, vx1, ax1, T)
        self.y_cubic = QuinticPolynomial(y0, vy0, ay0, y1, vy1, ay1, T)

    def calc_position(self, t):
        r"""
        calc position
        """
        x = self.x_cubic.calc_xt(t)
        y = self.y_cubic.calc_xt(t)

        return x, y

    def calc_curvature(self, t):
        r"""
        calc curvature
        """
        dx = self.x_cubic.calc_dxt(t)
        ddx = self.x_cubic.calc_ddxt(t)
        dy = self.y_cubic.calc_dxt(t)
        ddy = self.y_cubic.calc_ddxt(t)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)**(3/2)
        return k

    def calc_speed(self, t):
        r"""
        calc speed
        """
        dx = self.x_cubic.calc_dxt(t)
        dy = self.y_cubic.calc_dxt(t)
        v = math.hypot(dy, dx)
        return v

    def calc_yaw(self, t):
        r"""
        calc yaw
        """
        dx = self.x_cubic.calc_dxt(t)
        dy = self.y_cubic.calc_dxt(t)
        yaw = math.atan2(dy, dx)
        return yaw

    def calc_acc(self, t):
        r"""
        calc acceleration
        """
        ddx = self.x_cubic.calc_ddxt(t)
        ddy = self.y_cubic.calc_ddxt(t)
        return ddx, ddy


class Trajectory:
    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.a = []
        self.jerk = []
        self.k = []


def simulation_cubic():
    sx, sy, syaw, sv, sa = 10.0, 10.0, np.deg2rad(0.0), 4.0, 1.0
    gx, gy, gyaw, gv, ga = 30.0, -10.0, np.deg2rad(180.0), 4.0, 0.0

    MAX_ACCEL = 2.0  # max accel [m/s2]
    MAX_CURV = 1/2.0  # max curvature [1/m]
    dt = 0.1  # T tick [s]

    MIN_T = 5
    MAX_T = 100
    T_STEP = 5

    sv_x = sv * math.cos(syaw)
    sv_y = sv * math.sin(syaw)
    gv_x = gv * math.cos(gyaw)
    gv_y = gv * math.sin(gyaw)

    sa_x = sa * math.cos(syaw)
    sa_y = sa * math.sin(syaw)
    ga_x = ga * math.cos(gyaw)
    ga_y = ga * math.sin(gyaw)

    path = Trajectory()

    for T in np.arange(MIN_T, 100, T_STEP):
        path = Trajectory()
        cp = CubicPolynomial2D(sx, sv_x, gx, gv_x, sy, sv_y, gy, gv_y, T)

        for t in np.arange(0.0, T + dt, dt):
            path.t.append(t)
            x, y = cp.calc_position(t)
            path.x.append(x)
            path.y.append(y)

            v = cp.calc_speed(t)
            yaw = cp.calc_yaw(t)
            path.v.append(v)
            path.yaw.append(yaw)

            ax, ay = cp.calc_acc(t)
            a = np.hypot(ax, ay)
            path.a.append(a)

            if len(path.v) >= 2 and path.v[-1] - path.v[-2] < 0.0:
                a *= -1
            path.a.append(a)

            k = cp.calc_curvature(t)
            path.k.append(k)

        if max(np.abs(path.a)) <= MAX_ACCEL and max(np.abs(path.k)) <= MAX_CURV:
            break

    print("t_len: ", path.t, "s")
    print("max_v: ", max(path.v), "m/s")
    print("max_a: ", max(np.abs(path.a)), "m/s2")
    print(f"max_curvature: {max(np.abs(path.k))} 1/m")

    for i in range(len(path.t)):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.axis("equal")
        plt.plot(path.x, path.y, linewidth=2, color='gray')
        draw.Car(sx, sy, syaw, 1.5, 3)
        draw.Car(gx, gy, gyaw, 1.5, 3)
        draw.Car(path.x[i], path.y[i], path.yaw[i], 1.5, 3)
        plt.title(f"Cubic Polynomial Curves: speed {int(path.v[i]*10)/10} m/s")
        plt.grid(True)
        plt.pause(0.001)

    plt.show()


def simulation_quintic():
    sx, sy, syaw, sv, sa = 10.0, 10.0, np.deg2rad(0.0), 4.0, 1.0
    gx, gy, gyaw, gv, ga = 30.0, -10.0, np.deg2rad(180.0), 4.0, 0

    MAX_ACCEL = 2.0  # max accel [m/s2]
    MAX_CURV = 1/2.0  # max curvature [1/m]
    dt = 0.1  # T tick [s]

    MIN_T = 5
    MAX_T = 100
    T_STEP = 5

    sv_x = sv * math.cos(syaw)
    sv_y = sv * math.sin(syaw)
    gv_x = gv * math.cos(gyaw)
    gv_y = gv * math.sin(gyaw)

    sa_x = sa * math.cos(syaw)
    sa_y = sa * math.sin(syaw)
    ga_x = ga * math.cos(gyaw)
    ga_y = ga * math.sin(gyaw)

    path = Trajectory()

    for T in np.arange(MIN_T, 100, T_STEP):
        path = Trajectory()
        cp = QuinticPolynomial2D(
            sx, sv_x, sa_x, gx, gv_x, ga_x, sy, sv_y, sa_y, gy, gv_y, ga_y, T)

        for t in np.arange(0.0, T + dt, dt):
            path.t.append(t)
            x, y = cp.calc_position(t)
            path.x.append(x)
            path.y.append(y)

            v = cp.calc_speed(t)
            yaw = cp.calc_yaw(t)
            path.v.append(v)
            path.yaw.append(yaw)

            ax, ay = cp.calc_acc(t)
            a = np.hypot(ax, ay)
            path.a.append(a)

            if len(path.v) >= 2 and path.v[-1] - path.v[-2] < 0.0:
                a *= -1
            path.a.append(a)

            k = cp.calc_curvature(t)
            path.k.append(k)

        if max(np.abs(path.a)) <= MAX_ACCEL and max(np.abs(path.k)) <= MAX_CURV:
            break

    print("t_len: ", path.t, "s")
    print("max_v: ", max(path.v), "m/s")
    print("max_a: ", max(np.abs(path.a)), "m/s2")
    print(f"max_curvature: {max(np.abs(path.k))} 1/m")

    for i in range(len(path.t)):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.axis("equal")
        plt.plot(path.x, path.y, linewidth=2, color='gray')
        draw.Car(sx, sy, syaw, 1.5, 3)
        draw.Car(gx, gy, gyaw, 1.5, 3)
        draw.Car(path.x[i], path.y[i], path.yaw[i], 1.5, 3)

        plt.title(
            f"Quintic Polynomial Curves: speed {int(path.v[i]*10)/10} m/s")
        plt.grid(True)
        plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    # simulation_cubic()
    simulation_quintic()
