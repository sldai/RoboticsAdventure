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

<table>
<thead>
  <tr>
    <th></th>
    <th></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>![purepursuit](figure/pp.gif)</td>
    <td>![frontwheelfeedback](figure/frontwheel.gif)</td>
  </tr>
  <tr>
    <td>![rearwheelfeedback](figure/rearwheel.gif)</td>
    <td>![lqr kinematic](figure/lqr_kinematic.gif)</td>
  </tr>
</tbody>
</table>
