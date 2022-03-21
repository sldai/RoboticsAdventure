import numpy as np
from scipy.spatial.transform import Rotation as sci_Rot
from quad_model import QuadModel, params

def hat_map(x):
    a, b, c = x
    return np.array([[0,  -c,  b],
                     [c,   0, -a],
                     [-b,   a,  0]])

def vee_map(hat_x):
    return np.array([-hat_x[1, 2], hat_x[0, 2], -hat_x[0, 1]])


kp_x = 2.3
kp_y = 2.3
kp_z = 2.5
kd_x = 3.0
kd_y = 3.0
kd_z = 5.5

kp_phi = 30.0
kp_theta = 30.0
kp_psi = 10.0
kd_phi = 1.0
kd_theta = 1.0
kd_psi = 1.0


def se3_control(pos_t, vel_t, quat_t, omega_t, pos_ref, vel_ref, acc_ref, yaw_ref, yawdot_ref):
    r = pos_t
    r_dot = vel_t
    wRb = sci_Rot.from_quat(quat_t).as_matrix()
    omega = omega_t
    r_T = pos_ref
    r_T_dot = vel_ref
    r_T_ddot = acc_ref
    psi_T = yaw_ref
    psi_T_dot = yawdot_ref

    ep = r_T - r
    ev = r_T_dot - r_dot
    a_des = wRb@np.diag([kp_x, kp_y, kp_z])@wRb.T@ep + \
        wRb@np.diag([kd_x, kd_y, kd_z])@wRb.T@ev + r_T_ddot
    a_des = np.clip(a_des, -5, 5) # limit total acceleration

    # Thrust
    F_des = params.mass * (np.array([0, 0, params.g]) + a_des)
    F_des[2] = max(F_des[2], 0) # forbid move upside down
    thrust_des = F_des @ wRb[:, 2]

    # Moment
    Z_b_des = F_des/np.linalg.norm(F_des)
    X_c_des = np.array([np.cos(psi_T), np.sin(psi_T), 0])
    Y_b_des = np.cross(Z_b_des, X_c_des)
    Y_b_des /= np.linalg.norm(Y_b_des)
    X_b_des = np.cross(Y_b_des, Z_b_des)
    X_b_des /= np.linalg.norm(X_b_des)
    R_des = np.eye(3)
    R_des[:, 0] = X_b_des
    R_des[:, 1] = Y_b_des
    R_des[:, 2] = Z_b_des

    er = 1/2*vee_map(wRb.T@R_des - R_des.T@wRb) # R-R.T -> log(R)
    # reference angular velocity in body frame
    omega_ref = np.array([0, 0, psi_T_dot])
    ew = omega_ref - omega

    # get desired attitude, angular velocity, torque
    quat_des = sci_Rot.from_matrix(R_des).as_quat()
    omega_des = np.array([kp_phi, kp_theta, kp_psi])*er + omega_ref
    M_des = np.array([kp_phi, kp_theta, kp_psi])*er + \
        np.array([kd_phi, kd_theta, kd_psi]) * ew
    # return thrust_des, M_des
    return thrust_des, quat_des
    # return thrust_des, omega_des

from plot_utils import *
from traj_gen import CircleTrajectory, LineTrajectory


def track_line_trajectory():
    start_pos = np.array([2.0,9.0,7.0])
    start_attitude = sci_Rot.from_euler("xyz", [0,0,0]).as_quat()
    end_pos = np.array([-7.0, -8.0, -7.0])
    duration = 5.0
    rate = 100 # control rate
    dt = 1/rate

    quad_model = QuadModel(start_pos, start_attitude)

    traj_ref = LineTrajectory(start_pos-np.array([5,0,0]), end_pos, duration)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    for ind, t in enumerate(np.arange(0, duration+5, dt)):
        # reference state
        pos_ref = traj_ref.get_pos(t)
        vel_ref = traj_ref.get_vel(t)
        acc_ref = traj_ref.get_acc(t)
        yaw_ref = traj_ref.get_yaw(t)
        yawdot_ref = traj_ref.get_yawdot(t)

        # current state
        pos_cur = quad_model.position()
        vel_cur = quad_model.velocity()
        quat_cur = quad_model.attitude()
        omega_cur = quad_model.angular_velocity()

        thrust_des, omega_des = se3_control(pos_cur, vel_cur, quat_cur, omega_cur, pos_ref, vel_ref, acc_ref, yaw_ref, yawdot_ref)

        quad_model.update_FW(dt, thrust_des, omega_des)
        
        if ind % int(0.1/dt) == 0: # plot every 0.1 sec
            print("----------")
            print(params.maxF, thrust_des)
            plt.cla()
            plot_frame(pos_ref, sci_Rot.from_euler('z', yaw_ref).as_quat())
            plot_frame(pos_cur, quat_cur)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_zlim(-10, 10)
            plt.pause(0.1*dt)
    plt.show()

def track_circle_trajectory():
    start_pos = np.array([0.0,0.0,0.0])
    start_attitude = sci_Rot.from_euler("xyz", [0,0,0]).as_quat()
    
    traj_ref = CircleTrajectory(np.zeros(3), 1, 2)
        
    duration = 20.0
    rate = 100 # control rate
    dt = 1/rate
    quad_model = QuadModel(start_pos, start_attitude)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    traj_vis = np.array([traj_ref.get_pos(t) for t in np.linspace(0,np.pi*2/traj_ref.omega)])

    for ind, t in enumerate(np.arange(0, duration, dt)):
        # reference state
        pos_ref = traj_ref.get_pos(t)
        vel_ref = traj_ref.get_vel(t)
        acc_ref = traj_ref.get_acc(t)
        yaw_ref = traj_ref.get_yaw(t)
        yawdot_ref = traj_ref.get_yawdot(t)

        # current state
        pos_cur = quad_model.position()
        vel_cur = quad_model.velocity()
        quat_cur = quad_model.attitude()
        omega_cur = quad_model.angular_velocity()

        thrust_des, quat_des = se3_control(pos_cur, vel_cur, quat_cur, omega_cur, pos_ref, vel_ref, acc_ref, yaw_ref, yawdot_ref)

        quat_des_noise = quat_des + np.random.standard_normal()*0.02
        quat_des_noise /= np.linalg.norm(quat_des_noise)
        thrust_noise = thrust_des+np.random.uniform(-0.3,0.3)*params.g*params.mass
        quad_model.update_FQ(dt, thrust_des, quat_des)
        
        if ind % int(0.1/dt) == 0: # plot every 0.1 sec
            # print("----------")
            # print(pos_ref, vel_ref, acc_ref)
            # print(pos_cur, vel_cur, thrust_noise/params.mass*sci_Rot.from_quat(quat_cur).as_matrix()[:,2]-np.array([0,0,params.g]))      
            plt.cla()
            ax.plot3D(traj_vis[:,0], traj_vis[:,1], traj_vis[:,2], "-.k", alpha=0.5)
            # plot_frame(pos_ref, sci_Rot.from_euler('z', yaw_ref).as_quat())
            plot_hummingbird(pos_cur, quat_cur)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)
            plt.tight_layout()
            plt.pause(0.1*dt)
        # if abs(t-duration/2)<dt:
        #     quad_model.state[:3] += np.random.uniform(-3,3,3)
    plt.show()

track_circle_trajectory()
# track_line_trajectory()