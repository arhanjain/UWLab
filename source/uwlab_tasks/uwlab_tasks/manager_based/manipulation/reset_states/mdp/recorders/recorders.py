# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.recorder_manager import RecorderTerm

from typing import Sequence

class StableStateRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class DROIDJointPosActionRecorder(RecorderTerm):
    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        arm_joint_pos = self._env.scene["robot"].data.joint_pos[..., :7]
        gripper_action = self._env.action_manager.action[..., -1:]
        full_action = torch.cat([arm_joint_pos, gripper_action], dim=-1)
        return "action/droid_joint_pos_action", full_action


class ActionRecorder(RecorderTerm):
    # def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
    #     return "action", self._env.action_manager.action

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        return "action/raw", self._env.action_manager.action

class ObsRecorder(RecorderTerm):
    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        return "obs", self.to_cpu(self._env.obs_buf)

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        return "obs", self.to_cpu(self._env.obs_buf)

    def to_cpu(self, obs_buf: dict) -> dict:
        # many images remaining on GPU can cause memory issues
        new_obs_buf = {}
        for key in obs_buf.keys():
            if "vision" in key:
                new_obs_buf[key] = {}
                for k, v in obs_buf[key].items():
                    new_obs_buf[key][k] = v.detach().cpu()
            else:
                new_obs_buf[key] = obs_buf[key]
        return new_obs_buf

class AllStatesRecorder(RecorderTerm):
    """Recorder term that records state of every articulation and rigid body in the scene.
    
    For rigid objects, records root_state_w (position, quaternion, linear velocity, angular velocity).
    For articulations, records root_state_w, joint_pos, and joint_vel.
    """

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        return "states", self._capture_all_states(env_ids)

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        return "states", self._capture_all_states(None)

    
    def _capture_all_states(self, env_ids) -> dict:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)

        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return extract_env_ids_values(self._env.scene.get_state(is_relative=True))
    # def _capture_all_states(self, env_ids) -> dict:
    #     if env_ids is None:
    #         env_ids = slice(None)

    #     data = {}
        
    #     # Record rigid object states (root_state_w: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13)
    #     for name, asset in self._env.scene.rigid_objects.items():
    #         data[name] = {
    #             "root_state_w": asset.data.root_state_w[env_ids].clone(),
    #         }
        
    #     # Record articulation states (root_state_w, joint_pos, joint_vel)
    #     for name, asset in self._env.scene.articulations.items():
    #         data[name] = {
    #             "root_state_w": asset.data.root_state_w[env_ids].clone(),
    #             "joint_pos": asset.data.joint_pos[env_ids].clone(),
    #             "joint_vel": asset.data.joint_vel[env_ids].clone(),
    #         }
    
    #     return data


class GraspRelativePoseRecorder(RecorderTerm):
    """Recorder term that records relative position, orientation, and gripper joint states for grasp evaluation."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Configuration for which robot and object to track
        self.robot_name = cfg.robot_name
        self.object_name = cfg.object_name
        self.gripper_body_name = cfg.gripper_body_name

    def record_pre_reset(self, env_ids):
        """Record relative pose between object and gripper, plus gripper joint states before reset."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self._env.device)

        # Get robot articulation and object rigid body
        robot = self._env.scene[self.robot_name]
        obj = self._env.scene[self.object_name]

        # Get object pose (root pose contains position and orientation)
        obj_root_state = obj.data.root_state_w[env_ids]  # Shape: (num_envs, 13) - pos(3) + quat(4) + vel(6)
        obj_pos = obj_root_state[:, :3]  # Position
        obj_quat = obj_root_state[:, 3:7]  # Quaternion (w, x, y, z)

        # Get gripper body pose from the robot articulation
        # Find the gripper body index
        gripper_body_idx = None
        for idx, body_name in enumerate(robot.body_names):
            if self.gripper_body_name in body_name:
                gripper_body_idx = idx
                break

        # Get specific body pose
        gripper_pos = robot.data.body_state_w[env_ids, gripper_body_idx, :3]
        gripper_quat = robot.data.body_state_w[env_ids, gripper_body_idx, 3:7]

        # Calculate relative transform: T_gripper_in_object = T_object^{-1} * T_gripper
        relative_pos, relative_quat = math_utils.subtract_frame_transforms(obj_pos, obj_quat, gripper_pos, gripper_quat)

        # Get gripper joint states as dict mapping joint names to positions
        gripper_joint_pos = robot.data.joint_pos[env_ids].clone()
        gripper_joint_dict = {joint_name: gripper_joint_pos[:, i] for i, joint_name in enumerate(robot.joint_names)}

        # Prepare data to record
        grasp_data = {
            "relative_position": relative_pos,
            "relative_orientation": relative_quat,
            "gripper_joint_positions": gripper_joint_dict,
        }

        return "grasp_relative_pose", grasp_data
