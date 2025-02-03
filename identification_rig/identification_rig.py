#!/usr/bin/env python3


# Isaac Gym imports
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch

LEVER_LENGTH = "0_150" # 0_100, 0_150
MASS = "1" # 0_5, 1

def main():
    # 1. Parse arguments (Isaac Gym utility function)
    args = gymutil.parse_arguments(description="Isaac Gym Boilerplate Example")

    # 2. Acquire the Gym interface
    gym = gymapi.acquire_gym()

    # 3. Configure the simulation
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z  # Z-up coordinate system
    sim_params.dt = 1.0 / 60.0             # simulation timestep
    sim_params.substeps = 2               # physics substeps

    # You can switch to PhysX or Flex depending on your installation
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.use_gpu = True  # set to False if you don't have GPU support

    # 4. Create the simulation (use GPU device 0, graphics device 0, PhysX, etc.)
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise Exception("Failed to create sim")

    # 5. Add a ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # 6. Create a viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

    # 7. Create (or load) assets and environments
    #    We'll just make one environment here as an example
    envs = []
    num_envs = 1

    # The spacing below is how far apart multiple envs would be placed if you had more than one
    env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
    env_upper = gymapi.Vec3(1.0, 1.0, 1.0)

    # Path where your URDF or mesh files exist
    asset_root = "./assets"
    # franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    # asset_file = "awd/data/assets/mini2_bdx/mini2_bdx.urdf"
    asset_file = f"identification_rig_{LEVER_LENGTH}m_{MASS}kg/robot.urdf"

    # Setup asset options
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = False

    # Load the asset
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    for i in range(num_envs):
        # Create an environment
        env = gym.create_env(sim, env_lower, env_upper, num_envs)
        envs.append(env)

        # Create an actor (the robot)
        pose = gymapi.Transform()
        pose.p.x = 0.0
        pose.p.y = 0.0
        pose.p.z = 1.0

        # Add the robot to the environment
        # The last two parameters: "franka" is the name for the actor, and i is the index
        handle = gym.create_actor(
            env, asset, pose, "actor", i, 1
        )

    # 8. Main simulation loop
    while not gym.query_viewer_has_closed(viewer):
        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Update the viewer’s graphics
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # This will sync the simulation to match the desired dt, avoiding super-fast execution
        gym.sync_frame_time(sim)

    # 9. Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
