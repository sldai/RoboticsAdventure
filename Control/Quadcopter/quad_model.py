import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as sci_Rot
import scipy.integrate as integrate


class params:
    mass = 0.18  # kg
    g = 9.81  # m/s/s
    I = np.array([(0.00025, 0, 2.55e-6),
                  (0, 0.000232, 0),
                  (2.55e-6, 0, 0.0003738)])

    invI = np.linalg.inv(I)
    arm_length = 0.086  # meter
    height = 0.05
    minF = 0.0
    maxF = 2.0 * mass * g
    L = arm_length
    H = height
    km = 1.5e-9
    kf = 6.11e-8
    r = km / kf

    #  [ F  ]         [ F1 ]
    #  | M1 |  = A *  | F2 |
    #  | M2 |         | F3 |
    #  [ M3 ]         [ F4 ]
    A = np.array([[1,  1,  1,  1],
                  [0,  L,  0, -L],
                  [-L,  0,  L,  0],
                  [r, -r,  r, -r]])

    invA = np.linalg.inv(A)


class QuadModel:
    """ Quadcopter dynamic model

    state  - 1 dimensional vector but used as 13 x 1. [x, y, z, xd, yd, zd, qx, qy, qz, qw, p, q, r]
    where [qw, qx, qy, qz] is quternion and [p, q, r] are angular velocity in body frame.

    We support different controlling methods:
    F      - thrust output from controller
    M      - shape 3, moments output from controller

    acc    - acceleraton output from controller
    quat   - quaternion output from controller

    acc    - acceleration output from controller
    p, q, r- angular velocity in body frame output from controller

    params - system parameters struct, arm_length, g, mass, etc.
    """

    def __init__(self, pos, attitude) -> None:
        """Init pose

        Args:
            pos (array): x, y, z
            attitude (array): qx, qy, qz, qw
        """
        self.state = np.zeros(13)
        self.state[0] = pos[0]
        self.state[1] = pos[1]
        self.state[2] = pos[2]
        self.state[6] = attitude[0]
        self.state[7] = attitude[1]
        self.state[8] = attitude[2]
        self.state[9] = attitude[3]

    def position(self):
        return self.state[0:3]

    def velocity(self):
        return self.state[3:6]

    def attitude(self):
        return self.state[6:10]

    def angular_velocity(self):
        return self.state[10:13]

    def update_FM(self, dt, F, M):
        def state_dot(state, t, F, M):
            x, y, z, xdot, ydot, zdot, qx, qy, qz, qw, p, q, r = state
            quat = np.array([qx, qy, qz, qw])
            wRb = sci_Rot.from_quat(quat).as_matrix()

            # acceleration - Newton's second law of motion
            acc = 1.0 / params.mass * (wRb @ (np.array([0, 0, F]))
                                       - np.array([0, 0, params.mass * params.g]))

            # convert angular velocity into world frame
            wx, wy, wz = sci_Rot.from_quat(
                quat).as_matrix() @ np.array([p, q, r])
            # angular velocity - using quternion
            # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
            K_quat = 2.0  # this enforces the magnitude 1 constraint for the quaternion
            quaterror = 1.0 - quat@quat
            qdot = (1.0/2) * np.array([[0, -wz, wy, wx],
                                       [wz,   0, -wx, wy],
                                       [-wy,  wx,   0, wz],
                                       [-wx, -wy, -wz,  0]]).dot(quat) + K_quat * quaterror * quat

            # angular acceleration - Euler's equation of motion
            # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
            omega = np.array([p, q, r])
            pqrdot = params.invI @ (M.flatten() -
                                    np.cross(omega, params.I @ omega))

            state_dot = np.zeros(13)
            state_dot[0] = xdot
            state_dot[1] = ydot
            state_dot[2] = zdot
            state_dot[3] = acc[0]
            state_dot[4] = acc[1]
            state_dot[5] = acc[2]
            state_dot[6] = qdot[0]
            state_dot[7] = qdot[1]
            state_dot[8] = qdot[2]
            state_dot[9] = qdot[3]
            state_dot[10] = pqrdot[0]
            state_dot[11] = pqrdot[1]
            state_dot[12] = pqrdot[2]
            return state_dot
        # limit thrust and Moment
        L = params.arm_length
        r = params.r
        prop_thrusts = params.invA.dot(
            np.r_[np.array([[F]]), M.reshape((3, 1))])
        prop_thrusts_clamped = np.maximum(np.minimum(
            prop_thrusts, params.maxF/4), params.minF/4)
        F = np.sum(prop_thrusts_clamped)
        M = params.A[1:].dot(prop_thrusts_clamped)
        self.state = integrate.odeint(
            state_dot, self.state, [0, dt], args=(F, M))[1]

    def update_FQ(self, dt, F, quat: np.ndarray):
        def state_dot(state, t, F):
            pos = state[:3]
            vel = state[3:6]
            # acceleration - Newton's second law of motion
            quat = self.state[6:10]
            wRb = sci_Rot.from_quat(quat).as_matrix()
            acc = 1.0 / params.mass * (wRb @ (np.array([0, 0, F]))
                                       - np.array([0, 0, params.mass * params.g]))
            state_dot = np.zeros(len(state))
            state_dot[:3] = vel
            state_dot[3:6] = acc
            return state_dot
        self.state[6:10] = quat
        self.state = integrate.odeint(
            state_dot, self.state, [0, dt], args=(F,))[1]

    def update_FW(self, dt, F, omega: np.ndarray):
        def state_dot(state, t, F):
            x, y, z, xdot, ydot, zdot, qx, qy, qz, qw, p, q, r = state
            quat = np.array([qx, qy, qz, qw])
            wRb = sci_Rot.from_quat(quat).as_matrix()
            # acceleration - Newton's second law of motion
            acc = 1.0 / params.mass * (wRb @ (np.array([0, 0, F]))
                                       - np.array([0, 0, params.mass * params.g]))

            # convert angular velocity into world frame
            wx, wy, wz = sci_Rot.from_quat(
                quat).as_matrix() @ np.array([p, q, r])
            # angular velocity - using quternion
            # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
            K_quat = 2.0  # this enforces the magnitude 1 constraint for the quaternion
            quaterror = 1.0 - quat@quat
            qdot = (1.0/2) * np.array([[0, -wz, wy, wx],
                                       [wz,   0, -wx, wy],
                                       [-wy,  wx,   0, wz],
                                       [-wx, -wy, -wz,  0]]).dot(quat) + K_quat * quaterror * quat

            state_dot = np.zeros(13)
            state_dot[0] = xdot
            state_dot[1] = ydot
            state_dot[2] = zdot
            state_dot[3] = acc[0]
            state_dot[4] = acc[1]
            state_dot[5] = acc[2]
            state_dot[6] = qdot[0]
            state_dot[7] = qdot[1]
            state_dot[8] = qdot[2]
            state_dot[9] = qdot[3]
            return state_dot
        self.state[10:13] = omega
        self.state = integrate.odeint(
            state_dot, self.state, [0, dt], args=(F,))[1]


def test_kinematics():
    """Init a random rotation, then make the quadcopter rotate in a fixed angular velocity represented by axis angle. Then compare the simulated result with the ideal result to check the rotation kinematics simulation.
    """
    for _ in range(10):
        omega = np.random.uniform(-1, 1, 3)
        duration = 3
        init = sci_Rot.from_rotvec(np.random.uniform(-1, 1, 3))
        rot_g = sci_Rot.from_rotvec(
            omega*duration).as_matrix()@init.as_matrix()
        rot_sim = QuadModel(np.zeros(3), init.as_quat())
        dt = 0.1
        for _ in np.arange(0, duration, dt):
            rot_cur = sci_Rot.from_quat(rot_sim.attitude()).as_matrix()
            rot_sim.update_AW(dt, np.zeros(3), rot_cur.T @
                              omega)  # pure rotation
        rot_cur = sci_Rot.from_quat(rot_sim.attitude()).as_matrix()
        assert np.max(np.abs(rot_cur-rot_g)) < 1e-4, f"{rot_cur} \n{rot_g}"


def test_dynamics():
    """Test rotation dynamics
    """
    pass
