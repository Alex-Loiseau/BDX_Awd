import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=False, default="saved_obs.pkl")
args = parser.parse_args()


isaac_init_pos = np.array(
    [
        0.0,
        0.05,
        -0.63,
        1.368,
        -0.78,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -0.065,
        0.635,
        1.38,
        -0.79,
    ]
)

isaac_joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "left_antenna",
    "right_antenna",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

obses = pickle.load(open(args.data, "rb"))

num_dofs = 16
dof_poses = []  # (dof, num_obs)
actions = []  # (dof, num_obs)

for i in range(num_dofs):
    dof_poses.append([])
    actions.append([])
    for obs in obses:
        dof_poses[i].append(obs[3 : 3 + 16][i])# - isaac_init_pos[i])
        actions[i].append(obs[-(16 + 3) : -3][i])

# plot action vs dof pos

nb_dofs = len(dof_poses)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(actions[i * nb_cols + j], label="action")
        axs[i, j].plot(dof_poses[i * nb_cols + j], label="dof_pos")
        axs[i, j].legend()
        axs[i, j].set_title(f"{isaac_joints_order[i * nb_cols + j]}")

# set ylim between -1 and 2
for ax in axs.flat:
    ax.set_ylim([-1, 2])


fig.suptitle(f"{args.data}")
plt.show()

obses_names = [
    "projected_gravity 0",
    "projected_gravity 1",
    "projected_gravity 2",
    # dof pos
    "pos_" + str(isaac_joints_order[0]),
    "pos_" + str(isaac_joints_order[1]),
    "pos_" + str(isaac_joints_order[2]),
    "pos_" + str(isaac_joints_order[3]),
    "pos_" + str(isaac_joints_order[4]),
    "pos_" + str(isaac_joints_order[5]),
    "pos_" + str(isaac_joints_order[6]),
    "pos_" + str(isaac_joints_order[7]),
    "pos_" + str(isaac_joints_order[8]),
    "pos_" + str(isaac_joints_order[9]),
    "pos_" + str(isaac_joints_order[10]),
    "pos_" + str(isaac_joints_order[11]),
    "pos_" + str(isaac_joints_order[12]),
    "pos_" + str(isaac_joints_order[13]),
    "pos_" + str(isaac_joints_order[14]),
    "pos_" + str(isaac_joints_order[15]),
    # dof vel
    "vel_" + str(isaac_joints_order[0]),
    "vel_" + str(isaac_joints_order[1]),
    "vel_" + str(isaac_joints_order[2]),
    "vel_" + str(isaac_joints_order[3]),
    "vel_" + str(isaac_joints_order[4]),
    "vel_" + str(isaac_joints_order[5]),
    "vel_" + str(isaac_joints_order[6]),
    "vel_" + str(isaac_joints_order[7]),
    "vel_" + str(isaac_joints_order[8]),
    "vel_" + str(isaac_joints_order[9]),
    "vel_" + str(isaac_joints_order[10]),
    "vel_" + str(isaac_joints_order[11]),
    "vel_" + str(isaac_joints_order[12]),
    "vel_" + str(isaac_joints_order[13]),
    "vel_" + str(isaac_joints_order[14]),
    "vel_" + str(isaac_joints_order[15]),
    # foot contacts
    "left_foot_contact",
    "right_foot_contact",
    # action
    "action_" + str(isaac_joints_order[0]),
    "action_" + str(isaac_joints_order[1]),
    "action_" + str(isaac_joints_order[2]),
    "action_" + str(isaac_joints_order[3]),
    "action_" + str(isaac_joints_order[4]),
    "action_" + str(isaac_joints_order[5]),
    "action_" + str(isaac_joints_order[6]),
    "action_" + str(isaac_joints_order[7]),
    "action_" + str(isaac_joints_order[8]),
    "action_" + str(isaac_joints_order[9]),
    "action_" + str(isaac_joints_order[10]),
    "action_" + str(isaac_joints_order[11]),
    "action_" + str(isaac_joints_order[12]),
    "action_" + str(isaac_joints_order[13]),
    "action_" + str(isaac_joints_order[14]),
    "action_" + str(isaac_joints_order[15]),
    # commands
    "command 0",
    "command 1",
    "command 2",
]
# Now plot all obs raw in a grid on a new plot

# obses = [[56 obs at time 0], [56 obs at time 1], ...]

nb_obs = len(obses[0])
nb_rows = int(np.sqrt(nb_obs))
nb_cols = int(np.ceil(nb_obs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_obs:
            break
        axs[i, j].plot([obs[i * nb_cols + j] for obs in obses])
        axs[i, j].set_title(obses_names[i * nb_cols + j])

fig.suptitle(f"{args.data}")
plt.show()