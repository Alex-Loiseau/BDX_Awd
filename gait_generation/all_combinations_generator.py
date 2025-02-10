import subprocess
import os
import numpy as np

DX_RANGE = [-0.08, 0.03]
DY_RANGE = [-0.1, 0.1]
DTHETA_RANGE = [-0.1, 0.1]

if os.path.exists("tmp"):
    os.rmdir("tmp")
os.makedirs("tmp", exist_ok=True)

xy_granularity = 0.01
theta_granularity = 0.03

xy_step_size = int((DX_RANGE[1] - DX_RANGE[0]) / xy_granularity)
theta_step_size = int((DTHETA_RANGE[1] - DTHETA_RANGE[0]) / theta_granularity)

print(xy_step_size, theta_step_size)
all = xy_step_size* xy_step_size * theta_step_size
print("total nb of combinations: ", all)
exit()



dxs = np.arange(DX_RANGE[0], DX_RANGE[1], 0.005)
dys = np.arange(DY_RANGE[0], DY_RANGE[1], 0.01)
dthetas = np.arange(DTHETA_RANGE[0], DTHETA_RANGE[1], 0.01)

print(len(dxs), len(dys), len(dthetas), len(dxs)*len(dys)*len(dthetas))

# for dx in range(DX_RANGE[0], DX_RANGE[1], 0.01):
