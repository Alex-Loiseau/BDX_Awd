import pickle
import numpy as np
import matplotlib.pyplot as plt


voltages = pickle.load(open("voltages.pkl", "rb"))
# list of [[voltage1, voltage2, ...], [voltage1, voltage2, ...], ...]
# list of voltages in time for each joint


joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

num_dofs = 14
dof_voltages = []  # (dof, num_obs)

for i in range(num_dofs):
    dof_voltages.append([])
    for j in range(len(voltages)):
        dof_voltages[i].append(voltages[j][i])
    
# plot voltage vs time

nb_dofs = len(dof_voltages)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(dof_voltages[i * nb_cols + j], label="voltage")
        axs[i, j].set_title(joints_order[i * nb_cols + j])

plt.show()

