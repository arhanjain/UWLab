import numpy as np
import mediapy as mp
from openpi_client import websocket_client_policy, image_tools

class DroidJointPosClient():
    def __init__(self, host: str, port: int, open_loop_horizon: int) -> None:
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = open_loop_horizon

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Infer the next action from the policy in a server-client setup
        """
        both = None
        ret = {}
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            curr_obs = self._extract_observation(obs)

            self.actions_from_chunk_completed = 0
            exterior_image = image_tools.resize_with_pad(
                curr_obs["right_image"], 224, 224
            )
            wrist_image = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            request_data = {
                "observation/exterior_image_1_left": exterior_image,
                "observation/wrist_image_left": wrist_image,
                "observation/state": curr_obs["joint_position"],
                "prompt": instruction,
            }
            server_response = self.client.infer(request_data)
            self.pred_action_chunk = server_response["actions"]
            both = np.concatenate([exterior_image, wrist_image], axis=1)

        if return_viz and both is None:
            curr_obs = self._extract_observation(obs)
            both = np.concatenate(
                [
                    image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                    image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                ],
                axis=1,
            )

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        # if action[-1].item() > 0.5:
        #     action = np.concatenate([action[:-1], np.ones((1,))])
        # else:
        #     action = np.concatenate([action[:-1], np.zeros((1,))])
        # action = action[:-1]
        print(f"gripper: {action[-1]}")

        return action, both

    def _extract_observation(self, obs_dict):
        # Assign images
        right_image = obs_dict["vision"]["external_camera"].clone().detach().cpu().numpy()[0]
        wrist_image = obs_dict["vision"]["wrist_camera"].clone().detach().cpu().numpy()[0]


        # Capture proprioceptive state
        # robot_state = obs_dict["policy"]
        # joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        # gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]
        # joint_position = np.zeros(7)
        # gripper_position = np.zeros(1)
        joint_position = obs_dict["vision"]["joint_pos"].clone().detach().cpu().numpy()[0]

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
        }

INSTRUCTION = "Stack the green cube on the blue cube"
import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
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
import os
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict


import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


    # wrap for video recording
    import datetime
    log_dir = os.path.join("logs", "rsl_rl", "eval_openpi", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
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


    print("hello using droid pos client")
    client = DroidJointPosClient(host="127.0.1.1", port=8000, open_loop_horizon=8)
    print("initialized ")

    # reset environment
    # obs = env.get_observations()
    EPISODES_TO_RUN = 20
    obs, info = env.reset()
    episode = 0
    video = []
    while simulation_app.is_running():
        with torch.inference_mode():
            action, viz = client.infer(obs, INSTRUCTION, return_viz=True)
            video.append(viz)
            action = torch.tensor(action).unsqueeze(0).float()
            # env stepping
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated.any() or truncated.any():
                success = env.unwrapped.reward_manager.get_term_cfg("success").func(env, **env.unwrapped.reward_manager.get_term_cfg("success").params)[0]
                mp.write_video(os.path.join(log_dir, f"episode_{episode}_{success}.mp4"), video, fps=10)

                episode += 1
                video = []
                obs, info = env.reset()
                client.reset()

                if episode >= EPISODES_TO_RUN:
                    break


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
