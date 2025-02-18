import json
import os
import subprocess
import numpy as np
import argparse

def main(bdx_type, n):
    if bdx_type == "go_bdx":
        slow = 0.221
        medium = 0.336
        fast = 0.568
        dx_max = [0, 0.1]
        dy_max = [0, 0.1]
        dtheta_max = [0, 0.25]
    elif bdx_type == "mini_bdx":
        slow = 0.05
        medium = 0.1
        fast = 0.15
        dx_max = [0, 0.05]
        dy_max = [0, 0.05]
        dtheta_max = [0, 0.25]
    elif bdx_type == "mini2_bdx":
        slow = 0.05
        medium = 0.1
        fast = 0.15
        dx_max = [0, 0.05]
        dy_max = [0, 0.05]
        dtheta_max = [0, 0.25]
    else:
        raise ValueError("Invalid bdx_type. Choose either 'go_bdx' or 'mini_bdx'.")

    presets_dir = f"../awd/data/assets/{bdx_type}/placo_presets"
    tmp_dir = os.path.join(presets_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    preset_speeds = ["medium"]#, "fast"]

    if args.sweep:
        dxs = np.arange(args.min_sweep_x, args.max_sweep_x, args.sweep_xy_granularity)
        dys = np.arange(args.min_sweep_y, args.max_sweep_y, args.sweep_xy_granularity)
        dthetas = np.arange(args.min_sweep_theta, args.max_sweep_theta, args.sweep_theta_granularity)
        all_n = len(dxs) * len(dys) * len(dthetas)
    else:
        all_n = n

    script_path = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_path, "../recordings")

    print("")
    print(f"============== GENERATING {all_n} MOTION FILES ==============")
    print("")
    for i in range(all_n):
        # Randomly select a preset speed
        selected_speed = np.random.choice(preset_speeds)
        # Load the selected preset
        with open(os.path.join(presets_dir, f"{selected_speed}.json")) as file:
            data = json.load(file)

        if not args.sweep:
            # Modify dx, dy, dtheta randomly
            data["dx"] = round(np.random.uniform(dx_max[0], dx_max[1]) * np.random.choice([-1, 1]), 2)
            data["dy"] = round(np.random.uniform(dy_max[0], dy_max[1]) * np.random.choice([-1, 1]), 2)
            data["dtheta"] = round(np.random.uniform(dtheta_max[0], dtheta_max[1]) * np.random.choice([-1, 1]), 2)
        else:
            dx_idx = i % len(dxs)
            dy_idx = (i // len(dxs)) % len(dys)
            dtheta_idx = (i // (len(dxs) * len(dys))) % len(dthetas)

            data["dx"] = round(dxs[dx_idx], 2)
            data["dy"] = round(dys[dy_idx], 2)
            data["dtheta"] = round(dthetas[dtheta_idx], 2)
            print("========", "dx", data["dx"], "dy", data["dy"], "dtheta", data["dtheta"])


        tmp_preset = os.path.join(tmp_dir, f"{selected_speed}.json")
        with open(tmp_preset, 'w') as file:
            json.dump(data, file, indent=4)

        if bdx_type in ["mini_bdx", "mini2_bdx"]:
            subprocess.run(['python', "gait_generator.py", "--skip_warmup", "--preset", f"{tmp_preset}", "--name", f"{i}", f"--{bdx_type.split('_')[0]}"])
        else:
            subprocess.run(['python', "gait_generator.py", "--preset", f"{tmp_preset}", "--name", f"{i}"])


    speeds = []
    preset_names = []

    # Iterate through all JSON files in the directory
    for filename in os.listdir(default_output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(default_output_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Extract the relevant information
            placo_data = data.get("Placo", {})
            avg_x_vel = placo_data.get("avg_x_lin_vel", 0)
            avg_y_vel = placo_data.get("avg_y_lin_vel", 0)
            preset_name = placo_data.get("preset_name", "unknown")

            # Calculate the total speed
            total_speed = np.sqrt(avg_x_vel**2 + avg_y_vel**2)

            print(total_speed, preset_name)


        # if (preset_name == 'slow' and total_speed > slow) or \
        #     (preset_name == 'medium' and (total_speed <= slow or total_speed > fast)) or \
        #     (preset_name == 'fast' and total_speed <= medium):

        #     # delete the file
        #     os.remove(file_path)
        #     print(f"Deleted {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP data")
    parser.add_argument("--bdx_type", choices=["go_bdx", "mini_bdx", "mini2_bdx"], required=True, help="Type of BDX to generate data for")
    parser.add_argument("--num", type=int, default=100, help="Number of motion files to generate.")
    parser.add_argument("--sweep", action="store_true", help="Sweep through the dx, dy, dtheta values.")
    parser.add_argument("--min_sweep_x", type=float, default=-0.03, help="Minimum value of dx to sweep through.")
    parser.add_argument("--max_sweep_x", type=float, default=0.08, help="Maximum value of dx to sweep through.")
    parser.add_argument("--min_sweep_y", type=float, default=-0.03, help="Minimum value of dy to sweep through.")
    parser.add_argument("--max_sweep_y", type=float, default=0.03, help="Maximum value of dy to sweep through.")
    parser.add_argument("--min_sweep_theta", type=float, default=-0.1, help="Minimum value of dtheta to sweep through.")
    parser.add_argument("--max_sweep_theta", type=float, default=0.1, help="Maximum value of dtheta to sweep through.")
    parser.add_argument("--sweep_xy_granularity", type=float, default=0.02, help="Granularity of dx and dy values.")
    parser.add_argument("--sweep_theta_granularity", type=float, default=0.03, help="Granularity of dtheta")
    args = parser.parse_args()
    main(args.bdx_type, args.num)
