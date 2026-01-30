# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to render dataset episodes using state information from HDF5 file.

This script loads episodes from an HDF5 dataset and renders them by resetting
the environment to the initial states stored in the dataset, similar to how
MultiResetManager uses env.reset_to.

Usage:
    python render_dataset_states.py --hdf5_file rollout_dataset/dataset.hdf5 --task <task_name>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Render dataset episodes using state information.")
parser.add_argument("--hdf5_file", type=str, required=True, help="Path to the HDF5 dataset file.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="List of episode indices to render. If empty, renders all episodes.",
)
# parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
parser.add_argument(
    "--episode_duration",
    type=float,
    default=5.0,
    help="Duration to display each episode in seconds. Set to 0 to display indefinitely.",
)
parser.add_argument(
    "--write_observations",
    action="store_false",
    default=True,
    help="Compute and write vision observations back to the HDF5 file.",
)
parser.add_argument(
    "--obs_dump_dir",
    type=str,
    default=None,
    help=(
        "If set and write_observations is enabled, dump per-episode vision observations "
        "to this directory as .npz files instead of modifying the HDF5 file."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import h5py
import numpy as np
import time
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "env_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Render dataset episodes using state information."""
    # Check if dataset file exists
    if not os.path.exists(args_cli.hdf5_file):
        raise FileNotFoundError(f"Dataset file not found: {args_cli.hdf5_file}")

    # Load dataset
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.hdf5_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    # Get episode names
    episode_names = list(dataset_file_handler.get_episode_names())

    # Determine which episodes to render
    if args_cli.select_episodes:
        episode_indices = [i for i in args_cli.select_episodes if i < episode_count]
    else:
        episode_indices = list(range(episode_count))

    if not episode_indices:
        print("No valid episodes to render.")
        return

    print(f"Found {episode_count} episodes in dataset")
    print(f"Rendering {len(episode_indices)} episode(s)")

    # Get task name
    if args_cli.task is None:
        if env_name is not None:
            # Try to construct task name from env_name
            task_name = f"Isaac-{env_name}-v0"
        else:
            raise ValueError("Task name must be specified via --task or found in dataset.")
    else:
        task_name = args_cli.task

    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if hasattr(args_cli, "device") and args_cli.device is not None else env_cfg.sim.device

    # Disable recorders and terminations for rendering
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # Create environment
    print(f"Creating environment: {task_name}")
    env = gym.make(task_name, cfg=env_cfg, render_mode="human").unwrapped

    # Render episodes
    rendered_count = 0
    for episode_idx in episode_indices:
        if episode_idx >= episode_count:
            print(f"Skipping episode {episode_idx} (out of range)")
            continue

        episode_name = episode_names[episode_idx]
        print(f"\nRendering episode {episode_idx}: {episode_name}")

        try:
            # Load episode data
            episode_data = dataset_file_handler.load_episode(episode_name, env.device)

            # Access states directly from episode data
            # States are stored in episode_data.data["states"] 
            if not hasattr(episode_data, "data") or "states" not in episode_data.data:
                print(f"  Warning: No states found in episode '{episode_name}', skipping")
                continue
            
            states_data = episode_data.data["states"]
            
            # Find the number of steps by checking the shape of any state array
            num_steps = None
            if "articulation" in states_data:
                for asset_name, asset_data in states_data["articulation"].items():
                    if "joint_position" in asset_data:
                        value = asset_data["joint_position"]
                        if isinstance(value, torch.Tensor):
                            num_steps = value.shape[0]
                        elif isinstance(value, np.ndarray):
                            num_steps = value.shape[0]
                        else:
                            num_steps = len(value)
                        break
            
            if num_steps is None or num_steps == 0:
                print(f"  Warning: Could not determine number of states for episode '{episode_name}', skipping")
                continue
            
            print(f"  Found {num_steps} state(s) in episode")

            # Reset environment IDs
            if env_cfg.scene.num_envs == 1:
                env_ids = torch.tensor([0], device=env.device)
            else:
                env_ids = torch.arange(env_cfg.scene.num_envs, device=env.device)

            # Prepare for writing observations if requested
            vision_observations: dict[str, list] = {}
            if args_cli.write_observations:
                # Initialize structure to store vision observations
                vision_observations = {}

            # Render each state
            state_duration = args_cli.episode_duration / num_steps if args_cli.episode_duration > 0 else 0
            
            for step_idx in range(num_steps):
                state_name = "initial_state" if step_idx == 0 else f"state_{step_idx}"
                print(f"  Rendering {state_name} (step {step_idx}/{num_steps-1})...")
                
                # Extract state for this step
                state_dict = {}
                
                # Extract articulation states
                if "articulation" in states_data:
                    state_dict["articulation"] = {}
                    for asset_name, asset_data in states_data["articulation"].items():
                        state_dict["articulation"][asset_name] = {}
                        for key in ["joint_position", "joint_velocity", "root_pose", "root_velocity"]:
                            if key in asset_data:
                                value = asset_data[key]
                                if isinstance(value, torch.Tensor):
                                    state_dict["articulation"][asset_name][key] = value[step_idx:step_idx+1].to(env.device)
                                elif isinstance(value, np.ndarray):
                                    state_dict["articulation"][asset_name][key] = torch.from_numpy(value[step_idx:step_idx+1]).to(env.device)
                                else:
                                    state_dict["articulation"][asset_name][key] = torch.tensor([value[step_idx]], device=env.device)
                
                # Extract rigid object states
                if "rigid_object" in states_data:
                    state_dict["rigid_object"] = {}
                    for asset_name, asset_data in states_data["rigid_object"].items():
                        state_dict["rigid_object"][asset_name] = {}
                        for key in ["root_pose", "root_velocity"]:
                            if key in asset_data:
                                value = asset_data[key]
                                if isinstance(value, torch.Tensor):
                                    state_dict["rigid_object"][asset_name][key] = value[step_idx:step_idx+1].to(env.device)
                                elif isinstance(value, np.ndarray):
                                    state_dict["rigid_object"][asset_name][key] = torch.from_numpy(value[step_idx:step_idx+1]).to(env.device)
                                else:
                                    state_dict["rigid_object"][asset_name][key] = torch.tensor([value[step_idx]], device=env.device)
                
                # Reset environment to this state
                env.reset_to(state_dict, env_ids=env_ids, is_relative=True)

                # Reset velocities to zero (similar to MultiResetManager)
                if hasattr(env.scene, "robot"):
                    robot = env.scene["robot"]
                    robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)

                # Compute observations if requested
                if args_cli.write_observations:
                    # Update observations after reset
                    obs_buf = env.observation_manager.compute()
                    
                    # Extract vision observations from obs_buf
                    # if hasattr(env, "obs_buf") and "vision" in env.obs_buf:

                    for camera_name, camera_data in obs_buf["vision"].items():
                        # Convert to numpy and store (remove batch dimension for single env)
                        if env_cfg.scene.num_envs == 1:
                            obs_np = camera_data.detach().cpu().numpy()[0]  # Remove batch dim
                        else:
                            obs_np = camera_data.detach().cpu().numpy()
                        
                        # Initialize list for this camera if needed
                        if camera_name not in vision_observations:
                            vision_observations[camera_name] = []
                        
                        vision_observations[camera_name].append(obs_np)

                # Render this state
                start_time = time.time()
                try:
                    while simulation_app.is_running():
                        # Render the environment
                        env.sim.render()

                        # Check if duration limit reached for this state
                        if state_duration > 0:
                            elapsed = time.time() - start_time
                            if elapsed >= state_duration:
                                break
                except KeyboardInterrupt:
                    print(f"  Interrupted by user")
                    # Exit on Ctrl+C
                    break
                
                # Break outer loop if interrupted
                if not simulation_app.is_running():
                    break

            # Write or dump vision observations if requested
            if args_cli.write_observations and len(vision_observations) > 0:
                # If an obs dump directory is provided, write intermediate .npz instead of touching HDF5.
                if args_cli.obs_dump_dir is not None:
                    try:
                        os.makedirs(args_cli.obs_dump_dir, exist_ok=True)
                        out_path = os.path.join(args_cli.obs_dump_dir, f"{episode_name}_vision.npz")
                        # Stack lists into arrays for each camera
                        stacked = {}
                        for camera_name, obs_list in vision_observations.items():
                            obs_array = np.stack(obs_list, axis=0)
                            if obs_array.dtype != np.uint8:
                                if obs_array.max() <= 1.0:
                                    obs_array = (obs_array * 255).astype(np.uint8)
                                else:
                                    obs_array = obs_array.astype(np.uint8)
                            stacked[camera_name] = obs_array
                        np.savez_compressed(out_path, **stacked)
                        print(f"  Wrote intermediate observations to {out_path}")
                    except Exception as e:
                        print(f"  Error writing intermediate observations for episode {episode_name}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  Writing vision observations to HDF5...")
                    try:
                        # Close the dataset file handler first to release the file lock
                        dataset_file_handler.close()
                        
                        # Now open in read-write mode
                        with h5py.File(args_cli.hdf5_file, "r+") as h5f:
                            episode_path = f"data/{episode_name}"
                            if episode_path not in h5f:
                                print(f"  Warning: Episode path {episode_path} not found in HDF5, skipping write")
                            else:
                                # Create or access obs group
                                if "obs" not in h5f[episode_path]:
                                    h5f[episode_path].create_group("obs")
                                
                                obs_group = h5f[episode_path]["obs"]
                                
                                # Create or access vision group
                                if "vision" not in obs_group:
                                    obs_group.create_group("vision")
                                
                                vision_group = obs_group["vision"]
                                
                                # Write each camera's observations
                                for camera_name, obs_list in vision_observations.items():
                                    # Stack observations into array (num_steps, height, width, channels)
                                    obs_array = np.stack(obs_list, axis=0)
                                    
                                    # Convert to uint8 if needed (assuming images are in [0, 255] range)
                                    if obs_array.dtype != np.uint8:
                                        if obs_array.max() <= 1.0:
                                            obs_array = (obs_array * 255).astype(np.uint8)
                                        else:
                                            obs_array = obs_array.astype(np.uint8)
                                    
                                    # Write or overwrite the dataset
                                    if camera_name in vision_group:
                                        del vision_group[camera_name]
                                    vision_group.create_dataset(camera_name, data=obs_array, compression="gzip")
                                    
                                    print(f"    Wrote {camera_name}: shape {obs_array.shape}")
                                
                                h5f.flush()
                                print(f"  Successfully wrote vision observations for episode {episode_name}")
                        
                        # Reopen the dataset file handler for next episode
                        dataset_file_handler.open(args_cli.hdf5_file)
                    except Exception as e:
                        print(f"  Error writing observations to HDF5: {e}")
                        import traceback
                        traceback.print_exc()
                        # Try to reopen the dataset file handler even if write failed
                        try:
                            dataset_file_handler.open(args_cli.hdf5_file)
                        except Exception:
                            pass

            rendered_count += 1
            print(f"  Completed rendering episode {episode_idx}")

        except Exception as e:
            print(f"  Error rendering episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nFinished rendering {rendered_count} episode(s).")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

