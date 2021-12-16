import numpy as np

class Trajectory:
    def __init__(self) -> None:
        pass

    def get_pos(self, t):
        raise NotImplementedError()

    def get_vel(self, t):
        raise NotImplementedError()

    def get_acc(self, t):
        raise NotImplementedError()

    def get_jerk(self, t):
        raise NotImplementedError()

    def get_yaw(self, t):
        raise NotImplementedError()

    def get_yawdot(self, t):
        raise NotImplementedError()


class LineTrajectory(Trajectory):
    def __init__(self, x0, xT, duration) -> None:
        """Generate a polynomail line trajectory given start, end and time

        Args:
            x0 (np.ndarray): start pos
            xT (np.ndarray): end pos
            t (float): duration
        """
        super().__init__()

        self.x0 = np.array(x0)
        self.xT = np.array(xT)
        self.duration = duration
        self.degree = 5
        self.s_coeff = self.calc_poly_traj()

    def calc_poly_traj(self):
        """Calculate polynomial with pos, vel, acc boundary constraints
        """
        order = self.degree+1
        A = np.zeros([6, order])
        b = np.zeros(6)
        # pos constraints, s(0) = 0, s(T) = 1
        A[0] = self.get_poly_cc(order, 0, 0)
        b[0] = 0
        A[1] = self.get_poly_cc(order, 0, self.duration)
        b[1] = 1

        # vel constraints, s'(0) = 0, s'(T) = 0
        A[2] = self.get_poly_cc(order, 1, 0)
        b[2] = 0
        A[3] = self.get_poly_cc(order, 1, self.duration)
        b[3] = 0

        # acc constraints, s''(0) = 0, s''(T) = 0
        A[4] = self.get_poly_cc(order, 2, 0)
        b[4] = 0
        A[5] = self.get_poly_cc(order, 2, self.duration)
        b[5] = 0

        x = np.linalg.lstsq(A, b, rcond=None)[0]
        return x

    @staticmethod
    def get_poly_cc(n, k, t):
        """ This is a helper function to get the coeffitient of coefficient for n-th
            order (order=degree+1) polynomial with k-th derivative at time t.
        """
        assert (n > 0 and k >= 0), "order and derivative must be positive."

        cc = np.ones(n)
        D = np.linspace(0, n-1, n)

        for i in range(n):
            for j in range(k):
                cc[i] = cc[i] * D[i]
                D[i] = D[i] - 1
                if D[i] == -1:
                    D[i] = 0

        for i, c in enumerate(cc):
            cc[i] = c * np.power(t, D[i])

        return cc

    def get_pos(self, t):
        t = np.clip(t, 0, self.duration)
        s = self.get_poly_cc(self.degree+1, 0, t)@self.s_coeff
        x = self.x0 + s*(self.xT-self.x0)
        return x

    def get_vel(self, t):
        t = np.clip(t, 0, self.duration)
        sdot = self.get_poly_cc(self.degree+1, 1, t)@self.s_coeff
        xdot = sdot * (self.xT - self.x0)
        return xdot

    def get_acc(self, t):
        t = np.clip(t, 0, self.duration)
        sddot = self.get_poly_cc(self.degree+1, 2, t)@self.s_coeff
        xddot = sddot * (self.xT - self.x0)
        return xddot

    def get_yaw(self, t):
        diff = self.xT-self.x0
        return np.arctan2(diff[1], diff[0])

    def get_yawdot(self, t):
        return 0.0

class CircleTrajectory(Trajectory):
    def __init__(self, c, r, omega) -> None:
        """Calculate the circle trajectory

        Args:
            c (np.ndarray): center [x,y,z]
            r (float): radius
            omega (float): angular velocity
        """
        super().__init__()
        self.c = np.array(c)
        self.r = r
        self.omega = omega
    
    def get_pos(self, t):
        return self.r*np.array([np.cos(self.omega*t),np.sin(self.omega*t),0])+self.c

    def get_vel(self, t):
        return self.r*np.array([-np.sin(self.omega*t)*self.omega, np.cos(self.omega*t)*self.omega, 0])

    def get_acc(self, t):
        return self.r*np.array([-np.cos(self.omega*t)*self.omega**2, -np.sin(self.omega*t)*self.omega**2, 0])

    def get_yaw(self, t):
        psi = self.omega*t+np.pi/2*np.sign(self.omega)
        psi = psi % (np.pi*2)
        psi = psi if psi <=np.pi else psi-2*np.pi
        return psi

    def get_yawdot(self, t):
        return self.omega       