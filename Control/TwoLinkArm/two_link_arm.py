import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
import os

class param:
    I1 = 0.01
    I2 = 0.01
    m1 = 0.01
    m2 = 0.01
    l1 = 0.1
    l2 = 0.11
    r1 = l1/2
    r2 = l2/2

    dt = 0.01  # [s], control step size
    q_loop_step_num = 1  # step_num*dt step size for q loop
    x_loop_step_num = 5 # step_num*dt step size for x loop
    q2_range = [-3.0, 3.0]


def wrap_angle(angle):
    r"""Enforce the angle into [-\pi, \pi]
    """
    # angle = angle % (2*np.pi)
    # if angle >= np.pi:
    #     angle -= 2*np.pi
    return angle


class TwoLinkArm(object):
    r"""Simulate the two link arm with dynamics.
    Its configuration is defined as [\theta_1, \theta_2], 
    the end effector position [x, y] is calculated as:
    \begin{bmatrix}
    x\\ 
    y
    \end{bmatrix}
    =
    \begin{bmatrix}
    \cos(\theta_1) & \cos(\theta_1+\theta_2)\\ 
    \sin(\theta_1) & \sin(\theta_1+\theta_2)
    \end{bmatrix}
    \begin{bmatrix}
    l_1\\ 
    l_2
    \end{bmatrix}

    Its state is defined as [\theta_1, \theta_2, \dot{\theta_1}, \dot{\theta_2}].
    The Lagrange’s equation:
    \begin{bmatrix}
    \alpha + 2\beta c_2 & \delta + \beta c_2\\ 
    \delta + \beta c_2 & \sigma
    \end{bmatrix}
    \begin{bmatrix}
    \ddot{\theta}_1\\ 
    \ddot{\theta}_2
    \end{bmatrix}
    +
    \begin{bmatrix}
    -\beta s_2 \dot{\theta_2} & -\beta s_2(\dot{\theta}_1+\dot{\theta}_2)\\ 
    \beta s_2 \dot{\theta}_1 & 0
    \end{bmatrix}
    \begin{bmatrix}
    \dot{\theta}_1\\ 
    \dot{\theta}_2
    \end{bmatrix}
    =
    \begin{bmatrix}
    \tau_1\\ 
    \tau_2
    \end{bmatrix}
    """

    state_shape = 4
    q1_ind = 0
    q2_ind = 1
    q1_dot_ind = 2
    q2_dot_ind = 3

    action_shape = 2
    tau1_ind = 0
    tau2_ind = 1

    I1 = param.I1
    I2 = param.I2
    m1 = param.m1
    m2 = param.m2
    l1 = param.l1
    l2 = param.l2
    r1 = param.r1
    r2 = param.r2

    alpha = I1+I2+m1*r1**2+m2*(l1**2+r2**2)
    beta = m2*l1*r2
    delta = I2+m2*r2**2

    dt = param.dt

    def __init__(self):
        super().__init__()

    def reset(self):
        def r(): return np.random.uniform(-np.pi, np.pi)
        self.state = np.array([r(), r(), 0.0, 0.0])

    def state_dot(self, x, t, u):
        """Compute state derivative  
        """
        q1, q2, q1_dot, q2_dot = x
        tau1, tau2 = u

        alpha = self.alpha
        beta = self.beta
        delta = self.delta
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        D = np.array(
            [[alpha+2*beta*c2, delta+beta*c2],
             [delta+beta*c2, delta]]
        )

        C = np.array(
            [[-beta*s2*q2_dot, -beta*s2*(q1_dot+q2_dot)],
             [beta*s2*q1_dot, 0]]
        ) @ np.array([q1_dot, q2_dot])

        U = np.array([tau1, tau2])

        q_ddot = np.linalg.inv(D) @ (-C + U)
        state_dot = np.array([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])
        if q2 >= param.q2_range[1] or q2 <= param.q2_range[0]:
            state_dot[1] = 0
        return state_dot

    def enforce_bounds(self, state):
        state = state.copy()
        state[self.q1_ind] = wrap_angle(state[self.q1_ind])
        state[self.q2_ind] = wrap_angle(state[self.q2_ind])
        state[self.q2_ind] = np.clip(state[self.q2_ind], *param.q2_range)
        return state

    def propagate(self, start, control, duration):
        """Propagate dynamics
        """
        state = start.copy()
        state_next = odeint(self.state_dot, state, [
                            0, duration], args=(control,))[1]
        state_next = self.enforce_bounds(state_next)
        return state_next

    def step(self, control):
        assert len(control) == 2
        control = np.array(control, dtype=float)
        self.state = self.propagate(self.state, control, self.dt)
        return self.state

    def get_q1(self):
        """Return q1 and q1_dot
        """
        return self.state[0], self.state[2]

    def get_q2(self):
        """Return q2 and q2_dot
        """
        return self.state[1], self.state[3]

    def set_q1(self, q1):
        self.state[0] = q1

    def set_q2(self, q2):
        self.state[1] = q2


def getForwardModel(q1, q2):
    r"""    
    \begin{bmatrix}
    x\\ 
    y
    \end{bmatrix}
    =
    \begin{bmatrix}
    \cos(\theta_1) & \cos(\theta_1+\theta_2)\\ 
    \sin(\theta_1) & \sin(\theta_1+\theta_2)
    \end{bmatrix}
    \begin{bmatrix}
    l_1\\ 
    l_2
    \end{bmatrix}
    """
    x = param.l1 * np.cos(q1) + param.l2 * np.cos(q1+q2)
    y = param.l1 * np.sin(q1) + param.l2 * np.sin(q1+q2)
    return x, y

def getJacobian(q1, q2):
    l1 = param.l1
    l2 = param.l2

    J = np.array(
        [[-l1*np.sin(q1)-l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
         [l2*np.cos(q1)+l2*np.cos(q1+q2), l2*np.cos(q1+q2)]]
        )
    return J

def Jacobian_inv(q1, q2):
    J = getJacobian(q1, q2)
    u, s, vh = np.linalg.svd(J, full_matrices=True)
    sigma_max = 10
    s_inv = s.copy()
    for i in range(len(s)):
        if s[i] == 0.0:
            s_inv[i] = 10000.0
            continue
        if s[0]/s[i] > sigma_max:
            s_inv[i] = 10000.0
            continue
    for i in range(len(s_inv)):
        s_inv[i] = 1.0/s_inv[i]
    s_inv = np.diag(s_inv)

    J_inv = vh.T @ s_inv @ u.T
    return J_inv

def draw_pr2(q1, q2):
    plt.figure(figsize=(6, 6))
    origin = np.array([0.0, 0.0])
    point1 = np.array([param.l1*np.cos(q1), param.l1*np.sin(q1)])
    point2 = getForwardModel(q1, q2)
    line1 = np.linspace(origin, point1)
    line2 = np.linspace(point1, point2)
    plt.plot(line1[:, 0], line1[:, 1])
    plt.plot(line2[:, 0], line2[:, 1])
    plt.axis([-0.3, 0.3, -0.3, 0.3])


class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.001
        self.current_time = current_time
        self.last_time = self.current_time

        self.reset()

    def reset(self):
        """Clears PID computations and coefficients
        """

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, error, current_time):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """

        self.current_time = current_time
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + \
                (self.Ki * self.ITerm) + (self.Kd * self.DTerm)


def get_traj(theta):
    x = (0.19 + 0.02*np.cos(4*theta))*np.cos(theta)
    y = (0.19 + 0.02*np.cos(4*theta))*np.sin(theta)
    return x, y


def animate(state_seq):
    imgs = []
    plt.figure(figsize=(6, 6))
    for state in state_seq:
        plt.cla()
        q1, q2 = state[:2]
        origin = np.array([0.0, 0.0])
        point1 = np.array([param.l1*np.cos(q1), param.l1*np.sin(q1)])
        point2 = getForwardModel(q1, q2)
        line1 = np.linspace(origin, point1)
        line2 = np.linspace(point1, point2)
        plt.plot(line1[:, 0], line1[:, 1])
        plt.plot(line2[:, 0], line2[:, 1])
        plt.axis([-0.3, 0.3, -0.3, 0.3])
        plt.savefig('tmp.png')
        imgs.append(plt.imread('tmp.png'))
        
    os.remove('tmp.png')
    imageio.mimsave("pose.gif", imgs, duration=param.dt)

def configuration_control():
    env = TwoLinkArm()

    env.reset()  # should return a state vector if everything worked

    env.set_q1(0.0)
    env.set_q2(0.0)
    goal = np.array([np.pi/2, 0.0])

    controller_q1 = PID(P=0.1, I=0.0, D=0.3)
    controller_q2 = PID(P=0.1, I=0.0, D=0.3)

    seq = [env.state.copy()]
    t_seq = np.arange(start=0, stop=2+param.dt, step=param.dt)
    for i, t in enumerate(t_seq[:-1]):
        # calculate dq
        dq = goal - env.state[:2]
        e = dq

        # calculate dx
        controller_q1.update(e[0], t)
        controller_q2.update(e[1], t)

        dq = [controller_q1.output, controller_q2.output]

        # step
        env.step(dq)
        seq.append(env.state.copy())
    seq = np.array(seq)
    plt.plot(t_seq, seq[:, 0])
    plt.plot(t_seq, np.zeros(len(t_seq))+goal[0])
    plt.plot(t_seq, seq[:, 1])
    plt.plot(t_seq, np.zeros(len(t_seq))+goal[1])
    # draw ani
    animate(seq)
    plt.show()


def EED_position_control():
    env = TwoLinkArm()

    env.reset()  # should return a state vector if everything worked

    env.set_q1(0.0)
    env.set_q2(0.0)
    goal = np.array([0.0, 0.21]) # [x,y]

    # get dx, output substep dx
    controller_x = PID(P=0.3, I=0.0, D=0.)
    controller_y = PID(P=0.3, I=0.0, D=0.)

    # get dq, output torque
    controller_q1 = PID(P=0.1, I=0.0, D=0.3)
    controller_q2 = PID(P=0.1, I=0.0, D=0.3)

    seq = [env.state.copy()]
    t_seq = np.arange(start=0, stop=2+param.dt, step=param.dt)

    for i, t in enumerate(t_seq[:-1]):
        # calculate dx
        q1, q2 = env.state[:2]
        x, y = getForwardModel(q1, q2)
        e = goal - np.array([x,y])
        if i%param.x_loop_step_num == 0:
            controller_x.update(e[0], t)
            controller_y.update(e[1], t)
        dx = np.array([controller_x.output, controller_y.output])
        # calculate q target for the inner loop controlling q position
        q_target = Jacobian_inv(*env.state[:2]) @ dx + env.state[:2]
        e = q_target - env.state[:2]

        if i%param.q_loop_step_num == 0:
            # calculate torque
            controller_q1.update(wrap_angle(e[0]), t)
            controller_q2.update(wrap_angle(e[1]), t)

        torque = [controller_q1.output, controller_q2.output]

        # step
        env.step(torque)
        seq.append(env.state.copy())
    seq = np.array(seq)
    x_seq = np.array([getForwardModel(q1,q2) for q1,q2 in seq[:,:2]])
    plt.plot(t_seq, x_seq[:, 0])
    plt.plot(t_seq, np.zeros(len(t_seq))+goal[0])
    plt.plot(t_seq, x_seq[:, 1])
    plt.plot(t_seq, np.zeros(len(t_seq))+goal[1])
    # draw ani
    animate(seq)
    plt.show()    


if __name__ == "__main__":
    EED_position_control()
