# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint from a wandb run using RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import tempfile

import wandb

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent checkpoint from wandb using RSL-RL.")
parser.add_argument("--wandb_run_path", type=str, required=True, help="Wandb run path in format: entity/project/run_id")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def download_latest_checkpoint_from_wandb(wandb_run_path: str) -> str:
    """Download the latest checkpoint from a wandb run.
    
    Args:
        wandb_run_path: Path to wandb run in format "entity/project/run_id"
        
    Returns:
        Path to the downloaded checkpoint file
    """
    # Parse wandb run path
    parts = wandb_run_path.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid wandb run path format. Expected 'entity/project/run_id', got: {wandb_run_path}")
    entity, project, run_id = parts
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get the run
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"[INFO] Found wandb run: {run.name} ({run.id})")
    
    # Find checkpoint files - look for .pt files in model/ directory or files with checkpoint/model in name
    checkpoint_files = []
    for file in run.files():
        # Look for checkpoint files (typically .pt files)
        if file.name.endswith(".pt"):
            # Prioritize files in model/ directory or with checkpoint/model in path
            file_path_lower = file.name.lower()
            if (
                "model/" in file_path_lower
                or "checkpoint" in file_path_lower
                or file_path_lower.startswith("model")
                or file_path_lower.endswith("model.pt")
            ):
                checkpoint_files.append(file)
    
    # If no specific checkpoint files found, look for any .pt files
    if not checkpoint_files:
        checkpoint_files = [f for f in run.files() if f.name.endswith(".pt")]
    
    if not checkpoint_files:
        raise ValueError(
            f"No checkpoint files (.pt) found in wandb run {wandb_run_path}. "
            f"Available files: {[f.name for f in list(run.files())[:10]]}"
        )
    
    # Sort by name (assuming they have timestamps or version numbers) and get the latest
    # Also prefer files with "latest" or higher numbers in the name
    def sort_key(file):
        name_lower = file.name.lower()
        # Prefer files with "latest" in name
        if "latest" in name_lower:
            return (0, file.name)
        # Then prefer files with numbers (assuming they're versioned)
        import re
        numbers = re.findall(r"\d+", file.name)
        if numbers:
            return (1, -int(numbers[-1]), file.name)  # Negative for reverse sort
        return (2, file.name)
    
    checkpoint_files.sort(key=sort_key)
    latest_checkpoint = checkpoint_files[0]
    
    print(f"[INFO] Found {len(checkpoint_files)} checkpoint file(s), using: {latest_checkpoint.name}")
    
    # Create temporary directory for download
    download_dir = os.path.join(tempfile.gettempdir(), "wandb_checkpoints", run_id)
    os.makedirs(download_dir, exist_ok=True)
    
    # Download the checkpoint
    checkpoint_path = os.path.join(download_dir, latest_checkpoint.name)
    print(f"[INFO] Downloading checkpoint to: {checkpoint_path}")
    latest_checkpoint.download(replace=True, root=download_dir)
    
    return checkpoint_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent from wandb checkpoint."""
    # Download checkpoint from wandb
    resume_path = download_latest_checkpoint_from_wandb(args_cli.wandb_run_path)
    
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

