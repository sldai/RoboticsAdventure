import numpy as np
import matplotlib.pyplot as plt
from quad_model import params
from scipy.spatial.transform import Rotation as sci_Rot

class HummingBird:
    arm_length = params.arm_length
    prop_r = arm_length/3

    arm1 = np.array([[0, 0, 0], [arm_length, 0, 0]])
    arm2 = np.array([[0, 0, 0], [0, arm_length, 0]])
    arm3 = np.array([[0, 0, 0], [-arm_length, 0, 0]])
    arm4 = np.array([[0, 0, 0], [0, -arm_length, 0]])

    theta = np.linspace(0, np.pi*2, 20)
    circle = np.zeros([len(theta), 2])
    circle[:, 0] = prop_r * np.cos(theta)
    circle[:, 1] = prop_r * np.sin(theta)
    prop1 = np.zeros([len(theta), 3])
    prop2 = np.zeros([len(theta), 3])
    prop3 = np.zeros([len(theta), 3])
    prop4 = np.zeros([len(theta), 3])
    prop1[:, 0] = circle[:, 0] + arm_length
    prop1[:, 1] = circle[:, 1]
    prop2[:, 0] = circle[:, 0]
    prop2[:, 1] = circle[:, 1] + arm_length
    prop3[:, 0] = circle[:, 0] - arm_length
    prop3[:, 1] = circle[:, 1]
    prop4[:, 0] = circle[:, 0]
    prop4[:, 1] = circle[:, 1] - arm_length


def homogenize(x):
    return np.concatenate([x, np.ones([1, x.shape[1]])], axis=0)


def dehomogenize(x):
    return x[:3]/x[3]


def plot_hummingbird(pos: np.ndarray, attitude: np.ndarray):
    wTb = np.block([[sci_Rot.from_quat(attitude).as_matrix(), pos.reshape([3, 1])],
                    [np.zeros([1, 3]), np.ones([1, 1])]])
    arm1 = dehomogenize(wTb @ homogenize(HummingBird.arm1.T)).T
    arm2 = dehomogenize(wTb @ homogenize(HummingBird.arm2.T)).T
    arm3 = dehomogenize(wTb @ homogenize(HummingBird.arm3.T)).T
    arm4 = dehomogenize(wTb @ homogenize(HummingBird.arm4.T)).T
    prop1 = dehomogenize(wTb @ homogenize(HummingBird.prop1.T)).T
    prop2 = dehomogenize(wTb @ homogenize(HummingBird.prop2.T)).T
    prop3 = dehomogenize(wTb @ homogenize(HummingBird.prop3.T)).T
    prop4 = dehomogenize(wTb @ homogenize(HummingBird.prop4.T)).T

    ax = plt.gca()
    objs = [arm1, arm2, arm3, arm4, prop1, prop2, prop3, prop4]
    for obj in objs:
        ax.plot3D(obj[:, 0], obj[:, 1], obj[:, 2], 'b')


def plot_frame(pos: np.ndarray, quat: np.ndarray):
    rot = sci_Rot.from_quat(quat).as_matrix()
    x = rot[:, 0] + pos
    y = rot[:, 1] + pos
    z = rot[:, 2] + pos
    o = pos
    ax = plt.gca()
    ax.plot3D([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], 'r')
    ax.plot3D([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], 'g')
    ax.plot3D([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], 'b')
