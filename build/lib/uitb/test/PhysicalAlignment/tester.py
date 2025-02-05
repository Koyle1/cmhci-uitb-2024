import uitb
from uitb import test as uitb_test
import os

import numpy as np
import pandas as pd
# import mujoco
# import matplotlib.pyplot as plt

REMOTE_DISPLAY = True
###TODO: test with other OS than Ubuntu

if REMOTE_DISPLAY:
    process = os.popen("hostname --ip-address")
    _myipaddress = process.read().split()[0]
    process.close()

    # os.environ["DISPLAY"] = f"{_myipaddress}:1"  #run remotely (X server needs to be configured to accept remote connections, see https://askubuntu.com/a/34663)
    os.environ["DISPLAY"] = f":1"  #run remotely (X server needs to be configured to accept remote connections, see https://askubuntu.com/a/34663)

    process = os.popen("echo /run/user/$(id -u)")
    _pam = process.read().split("\n")[0]
    process.close()

    os.environ["XDG_RUNTIME_DIR"] = _pam
else:
    # use X11 forwarding (e.g., using MobaXterm)
    os.environ["DISPLAY"] = "localhost:10.0"  #forward to localhost:10.0

SIMULATOR_NAME = "unity_random_3ccr_1e1_v115"
USE_CLONED = False

RUN_PARAMETERS = {"action_sample_freq": 20,"evaluate": True}
USE_VR_CONTROLLER = True  #whether to use VR controller position or that of the welded body ##(the latter does not require calling mj_step, and thus results in the hypothetical reach envelope of the hand, ignoring the offset to the VR controller)

if "test_env" in locals():
    del test_env

simulator_path = os.path.join(os.path.dirname(uitb.__path__[0]), "simulators", SIMULATOR_NAME)
test_env = uitb.Simulator.get(simulator_path, run_parameters=RUN_PARAMETERS, render_mode="rgb_array", use_cloned=USE_CLONED)

welded_body = test_env.config["simulation"]["task"]["kwargs"]["right_controller_body"]
relpose = test_env.config["simulation"]["task"]["kwargs"]["right_controller_relpose"]
endeffector_name = "controller-right" if USE_VR_CONTROLLER else welded_body

video_output = True
num_episodes = 1
video_filename = f'/home/florian/uitb-sim2vr/user-in-the-box-private/uitb_distance_{endeffector_name}_test.mp4'
figure_filename = f'/home/florian/uitb-sim2vr/user-in-the-box-private/uitb_distance_{endeffector_name}_test.png'
table_filename = f'/home/florian/uitb-sim2vr/user-in-the-box-private/uitb_distance_{endeffector_name}_test.csv'
trajectories_table_columns = ['elv_angle_pos', 'shoulder_elv_pos'] + ['end-effector_xpos' + suffix for
                                                                        suffix in ('_x', '_y', '_z')]

uitb_test.ReachEnvelopeCheck(test_env,
                       USE_VR_CONTROLLER,
                       welded_body,
                       relpose,
                       endeffector_name,
                       trajectories_table_columns,
                       num_episodes,
                       video_output,
                       figure_filename,
                       table_filename,
                       video_filename
                       )



