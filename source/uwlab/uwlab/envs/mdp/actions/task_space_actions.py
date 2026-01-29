# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import omni.log
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class MultiConstraintDifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.MultiConstraintsDifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""

    def __init__(self, cfg: actions_cfg.MultiConstraintsDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name, preserve_order=True)
        # save only the first body index
        self._body_idx = body_ids
        self._body_name = body_names
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = [i - 1 for i in self._body_idx]
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the differential IK controller
        self._ik_controller = self.cfg.controller.class_type(
            cfg=self.cfg.controller, num_bodies=len(self._body_idx), num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.zeros((self.num_envs, len(self._body_idx), 3), device=self.device)
            self._offset_rot = torch.zeros((self.num_envs, len(self._body_idx), 4), device=self.device)
            for body_names, pose in self.cfg.body_offset.pose.items():
                offset_body_ids, offset_body_names = self._asset.find_bodies(body_names, preserve_order=True)
                offset_body_ids = [self._body_idx.index(i) for i in offset_body_ids]
                self._offset_pos[:, offset_body_ids] = torch.tensor(pose.pos, device=self.device)
                self._offset_rot[:, offset_body_ids] = torch.tensor(pose.rot, device=self.device)
        else:
            self._offset_pos, self._offset_rot = None, None

        self.enable_task_space_boundary = False
        if self.cfg.task_space_boundary is not None:
            task_space_boundary = torch.tensor(self.cfg.task_space_boundary, device=self.device)
            self.enable_task_space_boundary = True
            self.min_limits = task_space_boundary[:, 0]
            self.max_limits = task_space_boundary[:, 1]

        self.joint_pos_des = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def desired_joint_position(self):
        return self.joint_pos_des

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :][:, :, :, self._joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        B = len(self._jacobi_body_idx)
        rot_b = self._asset.data.root_link_quat_w
        rot_b_m = math_utils.matrix_from_quat(math_utils.quat_inv(rot_b))
        rot_b_m = rot_b_m.unsqueeze(1).expand(-1, B, -1, -1).reshape(-1, 3, 3)  # [N*B, 3, 3]

        jacobian_pos = jacobian[:, :, :3, :].view(-1, 3, self._num_joints)  # [N*B, 3, #joints]
        jacobian_rot = jacobian[:, :, 3:, :].view(-1, 3, self._num_joints)  # [N*B, 3, #joints]
        # multiply and reshape back
        jacobian[:, :, :3, :] = torch.bmm(rot_b_m, jacobian_pos).view(self.num_envs, B, 3, -1)
        jacobian[:, :, 3:, :] = torch.bmm(rot_b_m, jacobian_rot).view(self.num_envs, B, 3, -1)
        return jacobian

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
        if self.enable_task_space_boundary:
            self._ik_controller.ee_pos_des = self._ik_controller.ee_pos_des.clip(self.min_limits, self.max_limits)

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            self.joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            self.joint_pos_des = joint_pos.clone()
        # set the joint position command
        self._asset.set_joint_position_target(self.joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        num_body_idx = len(self._body_idx)
        ee_pose_w = self._asset.data.body_link_state_w[:, self._body_idx, :7].view(-1, 7)
        root_pose_w = self._asset.data.root_state_w[:, :7].repeat_interleave(num_body_idx, dim=0)
        # compute the pose of the body in the root frame
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b, ee_quat_b, self._offset_pos.view(-1, 3), self._offset_rot.view(-1, 4)
            )

        return ee_pos_b.view(-1, num_body_idx, 3), ee_quat_b.view(-1, num_body_idx, 4)

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b

        if self.cfg.body_offset is not None:
            # Flatten from (num_envs, num_bodies, 6, num_joints) → (num_envs * num_bodies, 6, num_joints)
            jacobian_flat = jacobian.reshape(-1, 6, jacobian.shape[3])  # (N*B, 6, num_joints)

            # Flatten offsets
            offset_pos_flat = self._offset_pos.reshape(-1, 3)  # (N*B, 3)
            offset_rot_flat = self._offset_rot.reshape(-1, 4)  # (N*B, 4)

            # 1) Translate part:    ṗ_link = ṗ_ee + ω̇_ee × r_(link←ee)

            #    => J_link_lin = J_ee_lin + -[r×]_offset * J_ee_ang
            row_lin = jacobian_flat[:, 0:3, :]  # (N*B, 3, num_joints)
            row_ang = jacobian_flat[:, 3:6, :]  # (N*B, 3, num_joints)

            skew = math_utils.skew_symmetric_matrix(offset_pos_flat)  # (N*B, 3, 3)

            # ṗ_link += -skew(r_offset) ⋅ ω̇_ee

            # We can do bmm: row_lin += bmm(-skew, row_ang)
            row_lin += torch.bmm(-skew, row_ang)

            # 2) Rotate part:    ω_link = R_offset ⋅ ω_ee
            R_offset = math_utils.matrix_from_quat(offset_rot_flat)  # (N*B, 3, 3)
            row_ang_new = torch.bmm(R_offset, row_ang)
            jacobian_flat[:, 3:6, :] = row_ang_new

            # Reshape back
            jacobian = jacobian_flat.view(self.num_envs, len(self._body_idx), 6, -1)
        return jacobian

class TransformedOneShotDifferentialIKAction(DifferentialInverseKinematicsAction):
    """One-shot Differential IK action term with coordinate frame transformation.

    Unlike standard DifferentialInverseKinematicsAction which recomputes IK at every physics
    step (e.g., 500 Hz), this action computes IK ONCE per policy step and holds the joint
    target fixed. This makes it equivalent to JointPos in terms of control structure,
    eliminating the distillation gap when training a JointPos student from an IK expert.

    The workflow is:

    1. Receive 6-DOF Cartesian commands [x, y, z, rx, ry, rz] in transformed frame
    2. Apply coordinate frame transformation to standard robot base frame
    3. Compute IK ONCE to get joint target (not recomputed during decimation)
    4. Hold joint target fixed for all physics steps (PD actuator tracks it)

    This exposes ``delta_joint_pos`` which is the relative joint position action - exactly
    what a JointPos policy would need to output to achieve the same behavior.
    """

    cfg: actions_cfg.TransformedOneShotDifferentialIKActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.TransformedOneShotDifferentialIKActionCfg, env: ManagerBasedEnv):
        # Initialize the parent IK action
        super().__init__(cfg, env)

        # Setup action root offset transformation
        if self.cfg.action_root_offset is not None:
            self._action_root_offset_pos = torch.tensor(cfg.action_root_offset.pos, device=self.device).repeat(
                self.num_envs, 1
            )
            self._action_root_offset_quat = torch.tensor(cfg.action_root_offset.rot, device=self.device).repeat(
                self.num_envs, 1
            )
        else:
            self._action_root_offset_pos = None
            self._action_root_offset_quat = None

        # Buffers for one-shot IK: joint target computed once per step
        self._joint_pos_des = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._delta_joint_pos = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def delta_joint_pos(self) -> torch.Tensor:
        return self._delta_joint_pos

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions: transform coordinates, apply scaling, compute IK, store joint target.

        Args:
            actions: The raw actions in shape (num_envs, 6) representing [x, y, z, rx, ry, rz].
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Transform actions from offset frame to standard frame (if offset is configured)
        actions_standard = self._transform_actions_to_standard_frame(actions)

        # Apply scaling and clipping
        self._processed_actions[:] = actions_standard * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        # Obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        # Set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

        # Compute IK ONCE and store the joint target (not recomputed in apply_actions)
        self._compute_joint_target(ee_pos_curr, ee_quat_curr, joint_pos)

    def apply_actions(self):
        """Apply the pre-computed joint position target to the articulation.

        Unlike the parent class which recomputes IK every physics step, this simply
        applies the joint target that was computed once in process_actions().
        """
        self._asset.set_joint_position_target(self._joint_pos_des, self._joint_ids)

    def _transform_actions_to_standard_frame(self, actions: torch.Tensor) -> torch.Tensor:
        """Transform actions from offset coordinate frame to standard robot base frame.

        Args:
            actions: The raw actions in offset frame, shape (num_envs, 6).

        Returns:
            The transformed actions in standard frame, shape (num_envs, 6).
        """
        if self._action_root_offset_pos is not None and self._action_root_offset_quat is not None:
            # Extract position and rotation deltas
            delta_pos_offset = actions[:, :3]  # [x, y, z]
            delta_rot_offset = actions[:, 3:6]  # [rx, ry, rz] in axis-angle

            # Get rotation matrix from offset-robot-base to standard-robot-base
            # The action_root_offset defines standard -> offset, so we need the inverse
            R_offset_to_standard = math_utils.matrix_from_quat(math_utils.quat_inv(self._action_root_offset_quat))

            # Transform position delta: rotate from offset coordinates to standard coordinates
            delta_pos_standard = torch.bmm(R_offset_to_standard, delta_pos_offset.unsqueeze(-1)).squeeze(-1)

            # Transform rotation delta (axis-angle): rotate the axis from offset coordinates to standard
            delta_rot_standard = torch.bmm(R_offset_to_standard, delta_rot_offset.unsqueeze(-1)).squeeze(-1)

            return torch.cat([delta_pos_standard, delta_rot_standard], dim=-1)
        else:
            return actions

    def _compute_joint_target(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, joint_pos: torch.Tensor):
        """Compute and store the joint position target using IK.

        Args:
            ee_pos: Current end-effector position in shape (num_envs, 3).
            ee_quat: Current end-effector orientation in shape (num_envs, 4).
            joint_pos: Current joint positions in shape (num_envs, num_joints).
        """
        if ee_quat.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            self._joint_pos_des[:] = self._ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
            self._delta_joint_pos[:] = self._joint_pos_des - joint_pos
        else:
            self._joint_pos_des[:] = joint_pos.clone()
            self._delta_joint_pos[:] = 0.0
