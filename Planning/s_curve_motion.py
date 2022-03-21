"""Ref: On Algorithms for planning S-curve Motion Profiles, Kim Doang Nguyen, Teck-Chew Ng, I-Ming Chen, 2008.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def pos_real_root(x):
    rs = np.roots(x)
    for r in rs:
        if r.imag==0 and r>=0:
            return r.real

class MotionProfile2:
    """Trapezoid trajectory, limit acceleration
    """
    def __init__(self, x_peak: np.ndarray):
        self.order = 2

        assert len(x_peak) == 3, f"{x_peak} is not valid [position, velocity, acceleration]"
        T = np.zeros(self.order+1)
        T[2] = pos_real_root([x_peak[2], T[1]*x_peak[2], -x_peak[0]])
        x_max = T[2]*x_peak[2]
        if T[2]*x_peak[2] > x_peak[1]:
            T[2] = pos_real_root([x_peak[2], -x_peak[1]])
        else:
            x_peak[1] = x_max
        T[1] = pos_real_root([T[2]*x_peak[2], T[2]**2*x_peak[2]-x_peak[0]])
        self.T = T
        self.x_peak = x_peak

    def get_T(self): return self.T[1] + 2*self.T[2]
    def get_pva(self, t: float) -> np.ndarray:
        """Given

        Args:
            t (float): _description_

        Returns:
            np.ndarray: [p,v,a]
        """
        p = 0
        v = 0
        a = 0
        if self.T[0] <= t <= self.T[2]:
            p = (t-(self.T[0]))**2*self.x_peak[2]/2
            v = (t-(self.T[0]))*self.x_peak[2]
            a = self.x_peak[2]
        elif self.T[2] <= t  <= self.T[1] + self.T[2]:
            p = (t-(self.T[2]))*self.x_peak[1] + self.T[2]**2*self.x_peak[2]/2
            v = self.x_peak[1]
            a = 0
        elif self.T[1] + self.T[2] <= t <= self.T[1] + 2*self.T[2]:
            p = -(t-(self.T[1]+self.T[2]))**2*self.x_peak[2]/2 + (t-(self.T[1]+self.T[2]))*self.T[2]*self.x_peak[2]+self.T[1]*self.x_peak[1]+self.T[2]**2*self.x_peak[2]/2
            v = -(t-(self.T[1]+self.T[2]))*self.x_peak[2] + self.T[2]*self.x_peak[2]
            a = -self.x_peak[2]
        else:
            p = self.x_peak[0]
            v = 0
            a = 0
        return np.array([p,v,a])


class MotionProfile3:
    """S-curve trajectory, limit jerk
    """
    def __init__(self, x_peak) -> None:
        self.order = 3

        assert len(x_peak) == 4, f"{x_peak} is not valid [position, velocity, acceleration, jerk]"
        T = np.zeros(self.order+1)
        T[3] = pos_real_root([2*x_peak[3], T[1]*x_peak[3]+3*T[2]*x_peak[3], T[1]*T[2]*x_peak[3] + T[2]**2*x_peak[3], -x_peak[0]])

        x_max = (2*T[2] + 2*T[3])*T[3]*x_peak[3]/2
        if x_max > x_peak[1]:
            T[3] = pos_real_root([x_peak[3], T[2]*x_peak[3], -x_peak[1]])
        else:
            x_peak[1] = x_max

        x_max = T[3]*x_peak[3]
        if x_max > x_peak[2]:
            T[3] = pos_real_root([x_peak[3], -x_peak[2]])
        else:
            x_peak[2] = x_max
        
        T[2] = pos_real_root([T[3]*x_peak[3], T[1]*T[3]*x_peak[3] + 3*T[3]**2*x_peak[3], T[1]*T[3]**2*x_peak[3] + 2*T[3]**3*x_peak[3] - x_peak[0]])
        x_max = (2*T[2] + 2*T[3])*T[3]*x_peak[3]/2
        if (x_max > x_peak[1]):
            T[2] = pos_real_root([T[3]*x_peak[3], T[3]**2*x_peak[3] - x_peak[1]])
        else:
            x_peak[1] = x_max

        T[1] = pos_real_root([T[2]*T[3]*x_peak[3] + T[3]**2*x_peak[3], T[2]**2*T[3]*x_peak[3] + 3*T[2]*T[3]**2*x_peak[3] + 2*T[3]**3*x_peak[3] - x_peak[0]])
        self.T = T
        self.x_peak = x_peak

    def get_T(self): return self.T[1] + 2*self.T[2] + 4*self.T[3]

    def get_pva(self, t: float) -> np.ndarray:
        p = 0
        v = 0
        a = 0
        if self.T[0] <=  t <= self.T[3]:
            p = (t-self.T[0])**3*self.x_peak[3]/6
            v = (t-self.T[0])**2*self.x_peak[3]/2
            a = (t-self.T[0])*self.x_peak[3]
        elif self.T[3] <= t <= self.T[2] + self.T[3]:
            p = (t-self.T[3])**2*self.x_peak[2]/2 + (t-self.T[3])*self.T[3]**2*self.x_peak[3]/2 + self.T[3]**3*self.x_peak[3]/6
            v = (t-self.T[3])*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2
            a = self.x_peak[2]
        elif self.T[2] + self.T[3] <= t <= self.T[2] + 2*self.T[3]:
            p = -(t-(self.T[2] + self.T[3]))**3*self.x_peak[3]/6 + (t-(self.T[2] + self.T[3]))**2*self.T[3]*self.x_peak[3]/2 + (t-(self.T[2] + self.T[3]))*(self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2) + self.T[2]**2*self.x_peak[2]/2 + self.T[2]*self.T[3]**2*self.x_peak[3]/2 + self.T[3]**3*self.x_peak[3]/6
            v = -(t-(self.T[2] + self.T[3]))**2*self.x_peak[3]/2 + (t-(self.T[2] + self.T[3]))*self.T[3]*self.x_peak[3] + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2)
            a = -(t-(self.T[2] + self.T[3]))*self.x_peak[3] + self.T[3]*self.x_peak[3]
        elif self.T[2] + 2*self.T[3] <= t <= self.T[1] + self.T[2] + 2*self.T[3]:
            p = (t-(self.T[2] + 2*self.T[3]))*self.x_peak[1] + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2)*self.T[3] + self.T[2]**2*self.x_peak[2]/2 + self.T[2]*self.T[3]**2*self.x_peak[3]/2 + self.T[3]**3*self.x_peak[3]/2
            v = self.x_peak[1]
            a = 0
        elif self.T[1] + self.T[2] + 2*self.T[3] <= t <= self.T[1] + self.T[2] + 3*self.T[3]:
            p = -(t-(self.T[1] + self.T[2] + 2*self.T[3]))**3*self.x_peak[3]/6 + (t-(self.T[1] + self.T[2] + 2*self.T[3]))*(self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]) + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2)*self.T[3] + self.T[1]*self.x_peak[1] + self.T[2]**2*self.x_peak[2]/2 + self.T[2]*self.T[3]**2*self.x_peak[3]/2 + self.T[3]**3*self.x_peak[3]/2
            v = -(t-(self.T[1] + self.T[2] + 2*self.T[3]))**2*self.x_peak[3]/2 + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3])
            a = -(t-(self.T[1] + self.T[2] + 2*self.T[3]))*self.x_peak[3]
        elif self.T[1] + self.T[2] + 3*self.T[3] <= t <= self.T[1] + 2*self.T[2] + 3*self.T[3]:
            p = -(t-(self.T[1] + self.T[2] + 3*self.T[3]))**2*self.x_peak[2]/2 + (t-(self.T[1] + self.T[2] + 3*self.T[3]))*(self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2) + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2)*self.T[3] + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3])*self.T[3] + self.T[1]*self.x_peak[1] + self.T[2]**2*self.x_peak[2]/2 + self.T[2]*self.T[3]**2*self.x_peak[3]/2 + self.T[3]**3*self.x_peak[3]/3
            v = -(t-(self.T[1] + self.T[2] + 3*self.T[3]))*self.x_peak[2] + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2)
            a = -self.x_peak[2]
        elif self.T[1] + 2*self.T[2] + 3*self.T[3] <= t <= self.T[1] + 2*self.T[2] + 4*self.T[3]:
            p = (t-(self.T[1] + 2*self.T[2] + 3*self.T[3]))**3*self.x_peak[3]/6 - (t-(self.T[1] + 2*self.T[2] + 3*self.T[3]))**2*self.T[3]*self.x_peak[3]/2 + (t-(self.T[1] + 2*self.T[2] + 3*self.T[3]))*self.T[3]**2*self.x_peak[3]/2 + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2)*self.T[2] + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3]/2)*self.T[3] + (self.T[2]*self.x_peak[2] + self.T[3]**2*self.x_peak[3])*self.T[3] + self.T[1]*self.x_peak[1] + self.T[2]*self.T[3]**2*self.x_peak[3]/2 + self.T[3]**3*self.x_peak[3]/3
            v = (t-(self.T[1] + 2*self.T[2] + 3*self.T[3]))**2*self.x_peak[3]/2 - (t-(self.T[1] + 2*self.T[2] + 3*self.T[3]))*self.T[3]*self.x_peak[3] + self.T[3]**2*self.x_peak[3]/2
            a = (t-(self.T[1] + 2*self.T[2] + 3*self.T[3]))*self.x_peak[3] - self.T[3]*self.x_peak[3]
        else:
            p = self.x_peak[0]
            v = 0
            a = 0
        return np.array([p,v,a])


def main():
    plt.subplot(1,2,1)
    traj2 = MotionProfile2([7.0, 3.0, 3.0])
    pos = []
    vel = []
    acc = []
    ts = np.linspace(0, traj2.get_T(), int(traj2.get_T()//0.01))
    for t in ts:
        pva = traj2.get_pva(t)
        pos.append(pva[0])
        vel.append(pva[1])
        acc.append(pva[2])
    plt.plot(ts, pos, label="position")
    plt.plot(ts, vel, label="velocity")
    plt.plot(ts, acc, label="acceleration")
    plt.legend()
    plt.title("Trapezoid Motion Profile")


    plt.subplot(1,2,2)
    traj3 = MotionProfile3([7.0, 3.0, 3.0, 10.0])
    pos = []
    vel = []
    acc = []
    ts = np.linspace(0, traj3.get_T(), int(traj3.get_T()//0.01))
    for t in ts:
        pva = traj3.get_pva(t)
        pos.append(pva[0])
        vel.append(pva[1])
        acc.append(pva[2])
    plt.plot(ts, pos, label="position")
    plt.plot(ts, vel, label="velocity")
    plt.plot(ts, acc, label="acceleration")
    plt.legend()
    plt.title("S-Curve Motion Profile")
    plt.show()



if __name__ == "__main__":
    main()