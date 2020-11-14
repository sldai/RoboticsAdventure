Overview
----------

This repository implements some common control algorithms, including PID, dynamic programming, LQR, MPC, and test them on
some kinematic/dynamic systems, including two link arm, invert pendulum, car, quadrotor.

Two link arm
----------

|                   <!-- -->                    |                         <!-- -->                         |                         <!-- -->                         |                        <!-- -->                                  |
| :--------------------------------------: | :-------------------------------------------------: | :----------------------------------------------: | :------------------------------------------------------: |
| ![PID_setpoint](figure/Setpoint_PID.gif) | ![PID_setpoint_IK](figure/Setpoint_PID_with_IK.gif) | ![PID_setpoint_IK](figure/PID_path_tracking.gif) | ![PID_setpoint_IK](figure/PID_path_tracking_with_IK.gif) |


Car
---------

|                                            |                                              |
|:------------------------------------------:|:--------------------------------------------:|
|        ![purepursuit](figure/pp.gif)       | ![frontwheelfeedback](figure/frontwheel.gif) |
| ![rearwheelfeedback](figure/rearwheel.gif) |  ![lqr kinematic](figure/lqr_kinematic.gif)  |
